# Distributed Hugging Face powered by fairscale
能够使用Pytorch分布式框架的Hugging face。

# Example
## LLaMA,LLaMA2,CodeLLaMA
```python
# torchrun --nproc_per_node gpu_num tmp.py
from d_hugging_face.model.llama import env_initialization,LlamaModel
from transformers import LlamaConfig
tp_size=4
pp_size=1
checkpoint_dir=None
env_initialization(tp_size, pp_size)

config = LlamaConfig.from_pretrained(checkpoint_dir)
config.tp_size=tp_size

model:LlamaModel = LlamaModel(config)

```