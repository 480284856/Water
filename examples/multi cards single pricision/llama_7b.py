import argparse
from model_loading.llama import load_llama,LLaMABenchmark,LLaMABenchmarkConfig
from models.llama import ModelArgs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_input_length", type=int, default=16)
    parser.add_argument("--max_output_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warm_up_nums", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--benchmark_times", type=int, default=1)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pipeline_size", type=int, default=1)
    benchmark_config:LLaMABenchmarkConfig=parser.parse_args()

    model_args_7b=ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=32000
    )
    llama_7b=load_llama(model_args_7b)

    benchmarkObject=LLaMABenchmark(
        model=llama_7b,
        benchmark_config=benchmark_config
    )

    benchmarkObject.launch_benchmark()
    print("llama_7b: {.3f}token/s".format(benchmarkObject.get_benchmark_result()))

if __name__ == "__main__":
    main()









