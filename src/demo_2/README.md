## bf16 implementation

vit with qwen2.5-tllm-bf16

llm with qwen2.5-tllm-bf16

### how to use

``` bash
python3 build_multimodal_engine.py \
    --model_type qwen2_5_vl \
    --model_path /sys/fs/cgroup/docker_install/code/qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir ./tmp/trt_engines/Qwen2_5-VL/fp16/1-gpu/vision/

python3 ../../qwen/convert_checkpoint.py \
    --model_dir=/sys/fs/cgroup/docker_install/code/qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir ./tmp/trt_engines/Qwen2_5-VL/fp16/1-gpu/ \
    --dtype bfloat16

trtllm-build --checkpoint_dir ./tmp/trt_engines/Qwen2_5-VL/fp16/1-gpu/ \
    --output_dir ./tmp/trt_engines/Qwen2_5-VL/fp16/1-gpu/llm/ \
    --gemm_plugin=bfloat16 \
    --gpt_attention_plugin=bfloat16 \
    --max_batch_size=4 \
    --max_input_len=3601 \
    --max_seq_len=4000 \
    --max_multimodal_len=14308 # (max_batch_size) * (num_visual_features)

python3 run.py \
    --hf_model_dir /sys/fs/cgroup/docker_install/code/qwen/Qwen2.5-VL-7B-Instruct \
    --engine_dir ./tmp/trt_engines/Qwen2_5-VL/fp16/1-gpu/

```

### package version

tensorrt_llm == 0.17.0.dev2024121700

onnx == 1.14.0

transformers == 4.49.0