import torch
import os
from models.llama2 import Transformer,ModelArgs


def load_llama(model_args: ModelArgs) -> Transformer:
    os.environ["RANK"]='0'
    os.environ["WORLD_SIZE"]='1'
    os.environ["MASTER_ADDR"]='localhost'
    os.environ["MASTER_PORT"]='12346'
    torch.distributed.init_process_group('nccl')

    # 把tensor的默认类型设置为GPU上的HalfTensor，这样就所有的tensor创建操作就会默认放到GPU上，而且是FP16精度
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model=Transformer(model_args)
    return model

def generate(model):
    input_ids=torch.tensor([1,2,3,4])
    start_attn_cache=0
    output=model.forward(input_ids, start_attn_cache)
    return output