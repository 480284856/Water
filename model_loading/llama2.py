import torch
import os
import sys
from models.llama2 import Transformer,ModelArgs
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def load_llama(model_args: ModelArgs)->Transformer:
    if not torch.distributed.is_initialized():
        if device == "cuda":
            torch.distributed.init_process_group("nccl")
        else:
            torch.distributed.init_process_group("gloo")
    if not model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

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

    model = Transformer(model_args)
    return model

def generate(model):
    input_ids=torch.tensor([1,2,3,4])
    start_attn_cache=0
    output=model.forward(input_ids, start_attn_cache)
    return output