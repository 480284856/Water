set -e
input_lengths=(16 64 256 1024)
output_lengths=(8 32 128 512)
batch_sizes=(1 2 4 8 16)
num_cards=(1 2 4)

model_name_or_path="Codefuse_13b"
echo "dp_size,i,o,b,token/s,tp_size,pp_size" > log_codefuse_7b.csv

# Assuming the two arrays have the same length
for j in $(seq 0 $((${#batch_sizes[@]}-1))); do
    for i in $(seq 0 $((${#input_lengths[@]} - 1))); do
        for n in $(seq 0 $((${#num_cards[@]} - 1)));do
            # fp16
            torchrun --nproc_per_node ${num_cards[$n]} codefuse_13b.py \
                    --model_path $model_name_or_path \
                    --max_input_length ${input_lengths[$i]} \
                    --max_output_length ${output_lengths[$i]} \
                    --batch_size ${batch_sizes[$j]} \
                    --warm_up_times 4 \
                    --device "cuda" \
                    --benchmark_times 10 \
                    --dp_size ${num_cards[$n]}
            sleep 3
        done
    done
done