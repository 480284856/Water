import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer
)

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
    def __init__(self, mode_name_or_path, benchmark_config: LLaMABenchmarkConfig) -> None:
        self.benchmark_config=benchmark_config
        self.device=self.benchmark_config.device if self.benchmark_config.device=='cpu' else f'{self.benchmark_config.device}:{int(os.environ.get("LOCAL_RANK", 0))}'

        self.tokenizer, self.model = self.env_initialization(mode_name_or_path)

        world_size=int(os.environ.get("WORLD_SIZE", 1))
        assert benchmark_config.dp_size*benchmark_config.tp_size*benchmark_config.pipeline_size == world_size

        self.speeds=[torch.tensor(0.0).cuda() for _ in range(world_size)]
        self.local_rank=int(os.environ.get("LOCAL_RANK", 0))

        HUMAN_ROLE_START_TAG = "<|role_start|>human<|role_end|>"
        BOT_ROLE_START_TAG = "<|role_start|>bot<|role_end|>"

        self.prompt = f"{HUMAN_ROLE_START_TAG}write a python function of quick sort.{BOT_ROLE_START_TAG}" 

    def launch_benchmark(self,):
        total_time_used=0

        self._warmup()
        for _ in range(self.benchmark_config.benchmark_times):
            time_used=self._generate()
            total_time_used+=time_used

        token_num=self.benchmark_config.max_output_length*self.benchmark_config.benchmark_times
        self.speed=torch.tensor(token_num/total_time_used).cuda()
    
    def get_benchmark_result(self,):
        dist.all_gather(self.speeds, self.speed)
        return sum(self.speeds)/(self.benchmark_config.tp_size*self.benchmark_config.pipeline_size)
 
    def _decode_one_token(self, input_ids, attention_mask):
        logits=self.model.forward(input_ids, attention_mask)

        logits_next=logits[0][:,-1]
        token_next=logits_next.argmax(dim=-1)
        input_ids=torch.concat([input_ids,token_next.view(-1,1)], dim=-1).to(self.device)
        
        return input_ids

    def _generate(self,):
        time_used=0
        
        inputs = self.tokenizer(self.prompt, return_tensors='pt', padding=True, add_special_tokens=False, max_length=self.benchmark_config.max_input_length, truncation=True).to("cuda")
        input_ids=inputs['input_ids']
        attention_mask=inputs['attention_mask']
        while(True):
            with torch.no_grad():
                start=time.perf_counter_ns()

                input_ids=self._decode_one_token(input_ids, attention_mask)

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

    def env_initialization(self, mode_name_or_path, benchmark_config) -> Tuple[LlamaTokenizer, LlamaForCausalLM]:
        if not torch.distributed.is_initialized():
            if benchmark_config.device == "cuda":
                torch.distributed.init_process_group("nccl")
            else:
                torch.distributed.init_process_group("gloo")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if benchmark_config.device == "cuda":
            torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")        

        tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<unk>")
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
        # try 4bit loading if cuda memory not enough
        model = AutoModelForCausalLM.from_pretrained(mode_name_or_path,
                                                    trust_remote_code=True,
                                                    load_in_4bit=True,
                                                    torch_dtype=torch.bfloat16)
        model.eval()

        return tokenizer, model

    def generate(self,):
        inputs = self.tokenizer(self.prompt, return_tensors='pt', padding=True, add_special_tokens=False).to("cuda")
        
        outputs = self.model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                top_p=0.95,
                temperature=0.1,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        gen_text = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(gen_text)

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

    benchmarkObject=LLaMABenchmark(
        mode_name_or_path=benchmark_config.model_path,
        benchmark_config=benchmark_config
    )

    benchmarkObject.launch_benchmark()
    print("llama_7b: {:3f}token/s".format(benchmarkObject.get_benchmark_result()))

if __name__ == "__main__":
    main()
