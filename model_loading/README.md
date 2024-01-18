# 此文件夹介绍如何加载不同参数量的模型
## 🚨注意事项
1. 模型配置参数中有两个关于cache kv的参数，这个参数最好不要默认，否则会额外占用非常大的显存
2. 所有的示例代码需要放在一个.py文件中，并使用torchrun运行。运行格式为：

    ```torchrun --nproc_node 【NUM GPU】 xx.py```
3. 默认使用所有的GPU进行TP。

## LLaMA2
```python
from model_loading.llama2 import load_llama
from models.llama2 import ModelArgs

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

model_args_70b=ModelArgs(
    dim=8192,
    multiple_of=4096,
    ffn_dim_multiplier=1.3,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    vocab_size=32000
)

model_args_7b_chat=ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    vocab_size=32000
)

model_args_13b_chat=ModelArgs(
    dim=5120,
    n_layers=40,
    n_heads=40,
    vocab_size=32000
)

model_args_70b_chat=ModelArgs(
    dim=8192,
    multiple_of=4096,
    ffn_dim_multiplier=1.3,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    vocab_size=32000
)

llama_7b=load_llama(model_args_7b)
llama_13b=load_llama(model_args_13b)
llama_70b=load_llama(model_args_70b)

llama_7b_chat=load_llama(model_args_7b_chat)
llama_13b_chat=load_llama(model_args_13b_chat)
llama_70b_chat=load_llama(model_args_70b_chat)
```

## CodeLLaMA
```python
from model_loading.codellama import load_llama
from models.codellama import ModelArgs

model_args_CodeLlama_7b = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-05,
    rope_theta=1000000,
    vocab_size=32016
)
model_args_CodeLlama_13b = ModelArgs(
    dim=5120,
    n_layers=40,
    n_heads=40,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-5,
    rope_theta=1000000,
    vocab_size=32016
)
model_args_CodeLlama_34b = ModelArgs(
    dim=8192,
    n_layers=48,
    n_heads=64,
    n_kv_heads=8,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-05,
    rope_theta=1000000,
    vocab_size=32000
)
model_args_CodeLlama_7b_Python = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-05,
    rope_theta=1000000,
    vocab_size=32000
)
model_args_CodeLlama_13b_Python = ModelArgs(
    dim=5120,
    n_layers=40,
    n_heads=40,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-5,
    rope_theta=1000000,
    vocab_size=32000
)
model_args_CodeLlama_34b_Python = ModelArgs(
    dim=8192,
    n_layers=48,
    n_heads=64,
    n_kv_heads=8,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-05,
    rope_theta=1000000,
    vocab_size=32000
)
model_args_CodeLlama_7b_Instruct = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-05,
    rope_theta=1000000,
    vocab_size=32016
)
model_args_CodeLlama_13b_Instruct = ModelArgs(
    dim=5120,
    n_layers=40,
    n_heads=40,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-5,
    rope_theta=1000000,
    vocab_size=32016
)
model_args_CodeLlama_34b_Instruct = ModelArgs(
    dim=8192,
    n_layers=48,
    n_heads=64,
    n_kv_heads=8,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-05,
    rope_theta=1000000,
    vocab_size=32000
)

CodeLlama_7b = load_llama(model_args_CodeLlama_7b)
CodeLlama_13b = load_llama(model_args_CodeLlama_13b)
CodeLlama_34b = load_llama(model_args_CodeLlama_34b)
CodeLlama_7b_Python = load_llama(model_args_CodeLlama_7b_Python)
CodeLlama_13b_Python = load_llama(model_args_CodeLlama_13b_Python)
CodeLlama_34b_Python = load_llama(model_args_CodeLlama_34b_Python)
CodeLlama_7b_Instruct = load_llama(model_args_CodeLlama_7b_Instruct)
CodeLlama_13b_Instruct = load_llama(model_args_CodeLlama_13b_Instruct)
CodeLlama_34b_Instruct = load_llama(model_args_CodeLlama_34b_Instruct)
```
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