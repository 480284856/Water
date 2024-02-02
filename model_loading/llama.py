import torch
import os
import sys
import time
from dataclasses import dataclass
from models.llama import Transformer,ModelArgs
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

def load_llama(model_args: ModelArgs, model_parallel_size=None, pipeline_length=1)->Transformer:
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

    model = Transformer(model_args)
    return model

@dataclass
class LLaMABenchmarkConfig:
    max_input_length:int
    max_output_length:int
    batch_size:int
    warm_up_nums:int
    benchmark_times:int
    device:str
    
    dp_size:int
    tp_size:int
    pipeline_size:int

class LLaMABenchmark():
    def __init__(self, model: Transformer, benchmark_config: LLaMABenchmarkConfig) -> None:
        self.model=model
        self.benchmark_config=benchmark_config
        
        world_size=os.environ.get("WORLD_SIZE", 1)
        assert benchmark_config.dp_size*benchmark_config.tp_size*benchmark_config.pipeline_size == world_size
        
        self.speeds=[0 for _ in range(benchmark_config.dp_size)]
        self.local_rank=os.environ.get("LOCAL_RANK", 0)

    def launch_benchmark(self,):
        total_time_used=0

        self._warmup()
        for _ in range(self.benchmark_config.benchmark_times):
            time_used=self._generate()
            total_time_used+=time_used

        token_num=self.benchmark_config.max_output_length*self.benchmark_config.benchmark_times
        self.speeds[self.local_rank]=token_num/total_time_used
    
    def get_benchmark_result(self,):
        return sum(self.speeds)/len(self.speeds)
 
    def _decode_one_token(self, input_ids):
        logits=self.model.forward(input_ids)

        logits_next=logits[0][:,-1]
        token_next=logits_next.argmax(dim=-1)
        input_ids=torch.concat([input_ids,token_next.view(-1,1)], dim=-1).to(self.benchmark_config.device)
        
        return input_ids

    def _generate(self,):
        time_used=0
        input_ids=torch.ones(self.benchmark_config.batch_size, self.benchmark_config.max_input_length, dtype=torch.long).to(self.benchmark_config.device)

        while(True):
            with torch.no_grad():
                start=time.perf_counter_ns()

                input_ids=self._decode_one_token(input_ids)

                end=time.perf_counter_ns()
                time_used+=((end-start)/1e9)

            if(input_ids.shape[1]>=self.benchmark_config.max_input_length+self.benchmark_config.max_output_length):
                break
        
        return time_used

    def _warmup(self,):
        while(self.benchmark_config.warm_up_times):
            self._generate()
            warm_up_times-=1

       