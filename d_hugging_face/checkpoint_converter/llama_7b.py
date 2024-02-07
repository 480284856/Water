import os
import torch
from transformers import LlamaConfig

def hf_checkpoint_converter(tp_size, saving_dir, single_card_ckpt_dir)->None:
    state_dict = dict()
    result=[dict() for _ in range(tp_size)]
    config:LlamaConfig=LlamaConfig.from_pretrained(single_card_ckpt_dir)

    for f in os.listdir(single_card_ckpt_dir):
        if not f.endswith(".bin"):
            continue
        sub_sd=torch.load(os.path.join(single_card_ckpt_dir, f), map_location='cpu')
        state_dict.update(sub_sd)

    # embedding
    key='model.embed_tokens.weight'
    weight=state_dict[key]
    for i in range(tp_size):
        result[i].update({
            key:torch.split(weight, weight.shape[-1]//tp_size, dim=-1)[i].clone()
        })

    # LlamaDecoderLayer
    # LlamaAttention
    key='model.layers.{}.self_attn.{}_proj.weight'
    key_input_layernorm='model.layers.{}.input_layernorm.weight'    
    key_post_attention_layernorm='model.layers.{}.post_attention_layernorm.weight'   

    key_gate_proj='model.layers.{}.mlp.gate_proj.weight'    
    key_up_proj='model.layers.{}.mlp.up_proj.weight'    
    key_down_proj='model.layers.{}.mlp.down_proj.weight'    

    for num_layer in range(config.num_hidden_layers):
        # attention
        for char in "qkv":
            k=key.format(num_layer, char)
            weight=state_dict[k]
            for i in range(tp_size):
                result[i].update({
                    key:torch.split(weight, weight.shape[0]//tp_size, dim=0)[i].clone()
                })
        
        k=key.format(num_layer, 'o')
        weight=state_dict[k]
        for i in range(tp_size):
            result[i].update({
                key:torch.split(weight, weight.shape[1]//tp_size, dim=1)[i].clone()
            })
        
        k=key_input_layernorm.format(num_layer)
        weight=state_dict[k]
        for i in range(tp_size):
            result[i].update({
                key:weight
            })

        k=key_post_attention_layernorm.format(num_layer)
        weight=state_dict[k]
        for i in range(tp_size):
            result[i].update({
                key:weight
            })

        # mlp
        k=key_gate_proj.format(num_layer)
        weight=state_dict[k]
        for i in range(tp_size):
            result[i].update({
                key:torch.split(weight, weight.shape[0]//tp_size, dim=0)[i].clone()
            })

        k=key_up_proj.format(num_layer)
        weight=state_dict[k]
        for i in range(tp_size):
            result[i].update({
                key:torch.split(weight, weight.shape[0]//tp_size, dim=0)[i].clone()
            })

        k=key_down_proj.format(num_layer)
        weight=state_dict[k]
        for i in range(tp_size):
            result[i].update({
                key:torch.split(weight, weight.shape[-1]//tp_size, dim=-1)[i].clone()
            })

    #norm
    key='model.norm.weight'
    weight=state_dict[key]
    for i in range(tp_size):
        result[i].update({
            key:weight
        })
    
    for i in range(tp_size):
        torch.save(result[i], os.path.join(saving_dir, f'model.bin.{i}'))