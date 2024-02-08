import os
import time
import torch
import argparse
import torch.distributed as dist
from dataclasses import dataclass
from transformers import LlamaConfig
from d_hugging_face.models.llama import env_initialization,LlamaModel

@dataclass
class LLaMABenchmarkConfig:
    model_path: str
    max_input_length:int
    max_output_length:int
    batch_size:int
    warm_up_times:int
    benchmark_times:int
    device:str
    
    dp_size:int
    tp_size:int
    pipeline_size:int

class LLaMABenchmark():
    def __init__(self, model: LlamaModel, benchmark_config: LLaMABenchmarkConfig) -> None:
        self.model=model
        self.benchmark_config=benchmark_config
        
        world_size=int(os.environ.get("WORLD_SIZE", 1))
        assert benchmark_config.dp_size*benchmark_config.tp_size*benchmark_config.pipeline_size == world_size
        
        self.speeds=[torch.tensor(0.0) for _ in range(world_size)]
        self.local_rank=int(os.environ.get("LOCAL_RANK", 0))

    def launch_benchmark(self,):
        total_time_used=0

        self._warmup()
        for _ in range(self.benchmark_config.benchmark_times):
            time_used=self._generate()
            total_time_used+=time_used

        token_num=self.benchmark_config.max_output_length*self.benchmark_config.benchmark_times
        self.speed=torch.tensor(token_num/total_time_used)
    
    def get_benchmark_result(self,):
        dist.all_gather(self.speeds, self.speed)
        return sum(self.speeds)/(self.benchmark_config.tp_size*self.benchmark_config.pipeline_size)
 
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
        warm_up_times=self.benchmark_config.warm_up_times
        while(warm_up_times):
            self._generate()
            warm_up_times-=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_input_length", type=int, default=16)
    parser.add_argument("--max_output_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warm_up_times", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--benchmark_times", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pipeline_size", type=int, default=1)
    benchmark_config:LLaMABenchmarkConfig=parser.parse_args()

    env_initialization(benchmark_config.tp_size, benchmark_config.pipeline_size)

    config=LlamaConfig.from_pretrained(benchmark_config.model_path)
    config.tp_size=benchmark_config.tp_size

    model:LlamaModel = LlamaModel(config)

    benchmarkObject=LLaMABenchmark(
        model=model,
        benchmark_config=benchmark_config
    )

    benchmarkObject.launch_benchmark()
    print("llama_7b: {:3f}token/s".format(benchmarkObject.get_benchmark_result()))

if __name__ == "__main__":
    main()







    

