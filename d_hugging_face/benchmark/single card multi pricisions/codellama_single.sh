set -e
input_lengths=(16 64 256 1024)
output_lengths=(8 32 128 512)
batch_sizes=(1 2 4 8 16)

model_name_or_path="CodeLlama-7b-Instruct-hf"
device="cuda:0"
# Assuming the two arrays have the same length
for j in $(seq 0 $((${#batch_sizes[@]}-1))); do
    for i in $(seq 0 $((${#input_lengths[@]} - 1))); do
                # fp16
                python codellama_single.py \
                        --model_name_or_path $model_name_or_path \
                        --max_input_length ${input_lengths[$i]} \
                        --max_output_length ${output_lengths[$i]} \
                        --batch_size ${batch_sizes[$j]} \
                        --warm_up_nums 10 \
                        --device $device
                sleep 3
                # int 8
                python codellama_single.py \
                        --model_name_or_path $model_name_or_path \
                        --max_input_length ${input_lengths[$i]} \
                        --max_output_length ${output_lengths[$i]} \
                        --batch_size ${batch_sizes[$j]} \
                        --warm_up_nums 10 \
                        --device $device \
                        --quantization_type "int8" \
                        --quantized_model_path CodeLlama-7b-Instruct-hf-param-int8 
                sleep 3
                # int 4
                python codellama_single.py \
                        --model_name_or_path $model_name_or_path \
                        --max_input_length ${input_lengths[$i]} \
                        --max_output_length ${output_lengths[$i]} \
                        --batch_size ${batch_sizes[$j]} \
                        --warm_up_nums 10 \
                        --device $device \
                        --quantization_type "int4" \
                        --quantized_model_path CodeLlama-7b-Instruct-hf-param-int4 
                sleep 3
        done
done