import torch.nn as nn
import torch
import os
import sys
import numpy as np
import random
import warnings
import torch.nn.functional as F
import math

from typing import Optional,Tuple
from transformers import LlamaModel,LlamaConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention,LlamaMLP,LLAMA_ATTENTION_CLASSES,LlamaFlashAttention2,LlamaSdpaAttention
from dataclasses import dataclass
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv
from transformers import GPTQConfig,AutoModelForCausalLM,AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

class LlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False, 
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False, 
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True, 
            init_method=lambda x: x
        )

class LlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int|None):
        super().__init__(config, layer_idx)
        self.n_local_head=config.num_attention_heads//config.tp_size
        self.n_local_key_value_heads=config.num_key_value_heads//config.tp_size

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            gather_output=False, 
            init_method=lambda x: x
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            gather_output=False, 
            init_method=lambda x: x
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            gather_output=False, 
            init_method=lambda x: x
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True,
            init_method=lambda x: x
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.n_local_head, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.n_local_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.n_local_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.n_local_head, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.n_local_key_value_heads * self.head_dim)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)

class LlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        del self.embed_tokens
        del self.layers

        self.embed_tokens=ParallelEmbedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers=nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])

        torch.cuda.empty_cache()

def env_initialization(model_parallel_size=None, pipeline_length=1)->None:
    if not torch.distributed.is_initialized():
        if device == "cuda":
            torch.distributed.init_process_group("nccl")
        else:
            torch.distributed.init_process_group("gloo")
    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size, pipeline_length)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device == "cuda":
        torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    # support for mac
    # 把tensor的默认类型设置为GPU上的HalfTensor，这样就所有的tensor创建操作就会默认放到GPU上，而且是FP16精度
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.HalfTensor)