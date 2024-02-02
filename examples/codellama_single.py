import os
import torch
import torch._dynamo
import dill
import time
import argparse  
from apex import amp
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, AutoConfig

def _model2torch_compile(model, device="cuda:0"):
    model=model.to(device)
    model.eval()

    torch._dynamo.reset()
    compiled_model=torch.compile(model)

    return compiled_model

def quantization(model_name_or_path, bit, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
    gptq_config = GPTQConfig(bits=bit, dataset=dataset, tokenizer=tokenizer, use_exllama=False)
    quantized_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=gptq_config, device_map='auto')
    save_path=model_name_or_path+"-int"+str(bit)
    quantized_model.save_pretrained(save_path)
    
    if( not args.disable_torch_compile ):
        print("[INFO]:convert the model to torch script module")
        return _model2torch_compile(quantized_model, device)
    else:
        print("[INFO]: using vanilla hf model")
        return quantized_model

def load_model_fp16(args):
    inputs=torch.randint(0,512,(2,128))
    attention_mask=torch.ones_like(inputs)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto')
    model=model.to(args.device).half()
    inputs=inputs.to(args.device)
    attention_mask=attention_mask.to(args.device)
    model.eval()

    if( not args.disable_torch_compile ):
        print("[INFO]:compiling model with torch.compile")
        compiled_model=_model2torch_compile(model, args.device)
    else:
        print("[INFO]: using vanilla hf model")
        compiled_model=model
    return compiled_model

def generate(model, args):
    '''
    inputs: (input_ids, attention_mask)
    '''
    input_ids=torch.ones(args.batch_size, args.max_input_length, dtype=torch.long).to(args.device)
    attention_mask=torch.ones_like(input_ids, dtype=torch.long).to(args.device)
    model.to(args.device)
    model.eval()
    time_used=0
    warm_up_nums=0 if not args.warm_up_nums else args.warm_up_nums
    
    def _warmup(args, warm_up_times, model):
        input_ids=torch.ones(args.batch_size, args.max_input_length, dtype=torch.int64).to(args.device)
        attention_mask=torch.ones_like(input_ids, dtype=torch.int64).to(args.device)
        while(warm_up_times):
            with torch.no_grad():
                if(not args.disable_torch_compile):
                    logits=model.forward(input_ids)
                    logits_next=logits[0][:,-1]
                    token_next=logits_next.argmax(dim=-1)
                    input_ids=torch.concat([input_ids,token_next.view(-1,1)], dim=-1).to(args.device)
                    attention_mask=torch.ones_like(input_ids, dtype=torch.long).to(args.device)
                else:
                    model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_output_length,
                        do_sample=False
                    )

                warm_up_times-=1
                
    if(not args.disable_torch_compile): # my implementation
        _warmup(args, warm_up_nums, model)

        print("[INFO]: generate using your implementation")
        input_ids=torch.ones(args.batch_size, args.max_input_length, dtype=torch.long).to(args.device)
        attention_mask=torch.ones_like(input_ids, dtype=torch.long).to(args.device)
        while(True):
            with torch.no_grad():
                start=time.time()

                logits=model.forward(input_ids)

                logits_next=logits[0][:,-1]
                token_next=logits_next.argmax(dim=-1)
                input_ids=torch.concat([input_ids,token_next.view(-1,1)], dim=-1).to(args.device)
                attention_mask=torch.ones_like(input_ids, dtype=torch.long).to(args.device)

                end=time.time()
                time_used+=(end-start)
            if(input_ids.shape[1]>=args.max_input_length+args.max_output_length):
                break
    else: # use hugging face's .generate() method
        _warmup(args, warm_up_nums, model)

        print("[INFO]: generate using .generate() from hugging face")
        start=time.time()
        model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_output_length,
            min_length=args.max_output_length,
            do_sample=False
        )
        end=time.time()
        time_used=end-start

    return args.max_output_length*args.batch_size/time_used

# if .safetensor file does exist, create it with random parameter
def safetensorChecker(args):
    def _have_safetensors(model_name_or_path):
        # 获取 'codellama' 目录下的所有文件
        files = os.listdir(model_name_or_path)

        # 检查是否存在以 '.safetensors' 结尾的文件
        has_safetensors = False
        for file in files:
            if file.endswith('.safetensors'):
                has_safetensors = True
                break
        return has_safetensors
    
    if( not _have_safetensors(args.model_name_or_path) ):
        print("[WARNING]:safetensors file not found, initilize model with random parameters")
        config=AutoConfig.from_pretrained(args.model_name_or_path)
        model=AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        saving_path=args.model_name_or_path+"-param"
        print("[WARNING]:save model with random parameters to "+saving_path)
        model.save_pretrained(saving_path)
        tokenizer.save_pretrained(saving_path)
        args.model_name_or_path=args.model_name_or_path+"-param"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_input_length", type=int, default=16)
    parser.add_argument("--max_output_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warm_up_nums", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quantization_type", choices=["int4", "int8"], default=None)
    parser.add_argument("--quantized_model_path", type=str, default=None, help="saved quantized model path")
    parser.add_argument("--disable_torch_compile", action="store_true")
    args=parser.parse_args()

    # precision conversion
    print("[INFO]:Start model convertion")
    if not args.quantization_type: # without quantization
        print("[INFO]:Start converting without quantization(with fp16)")
        safetensorChecker(args)
        model=load_model_fp16(args)
    else:
        # have not saved quantized model, so create it
        if(not os.path.exists(args.quantized_model_path)):
            safetensorChecker(args)
            if args.quantization_type=='int4':
                print("[INFO]:Start converting with int4 quantization")
                model=quantization(
                    model_name_or_path=args.model_name_or_path,
                    bit=4,
                    device=args.device
                )
            elif args.quantization_type=='int8':
                print("[INFO]:Start converting with int8 quantization")
                model=quantization(
                    model_name_or_path=args.model_name_or_path,
                    bit=8,
                    device=args.device
                )
        else: # quantized model dir is offered, use it
            print("[INFO]: use saved quantized model at: ", args.quantized_model_path)
            model = AutoModelForCausalLM.from_pretrained(args.quantized_model_path, device_map='auto').to(args.device)

    # benchmark
    print("[INFO]:Start benchmarking")
    token_ps= generate(
        model=model,
        args=args
    )

    print("[Benchmark Result]: Codellama-7B, max_input_length, max_output_length, batch_size, precision, token_ps")
    p=args.quantization_type if args.quantization_type else "fp16"
    print(args.max_input_length, args.max_output_length, args.batch_size, p, token_ps)