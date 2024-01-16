# 此文件夹介绍如何加载不同参数量的模型

## LLaMA
```python
from model_loading.llama import load_llama
from models.llama import ModelArgs

model_args_7b=ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    vocab_size=32000
)

model_args_13b=ModelArgs(
    dim=5120,
    n_layers=40,
    n_heads=40,
    vocab_size=32000
)
# 有待确认
model_args_30b=ModelArgs(
    dim=6656,
    n_layers=60,
    n_heads=52,
    vocab_size=32000
)
# 有待确认
model_args_70b=ModelArgs(
    dim=8192,
    n_layers=80,
    n_heads=64,
    vocab_size=32000
)

llama_7b=load_llama(model_args_7b)
llama_13b=load_llama(model_args_13b)
llama_30b=load_llama(model_args_30b)
llama_70b=load_llama(model_args_70b)
```