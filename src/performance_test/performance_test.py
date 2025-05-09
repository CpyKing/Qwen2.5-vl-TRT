from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import time

def torch_torch():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/sys/fs/cgroup/docker_install/code/qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # attn_implementation="sdpa",
        # attn_implementation="eager",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained("/sys/fs/cgroup/docker_install/code/qwen/Qwen2.5-VL-7B-Instruct")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/sys/fs/cgroup/zzz_tmp/my_dataset/images/womandog.jpeg",
                },
                {"type": "text", "text": "你是一个智能助手，我是用户，请识别这张图片并描述这张图片展示了什么内容。"},
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    for i in range(len(image_inputs)):
        image_inputs[i] = image_inputs[i].resize((1372, 2044))
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda").to(torch.bfloat16)
    
    # Inference: Generation of the output
    for i in range(2):
        e2e_start = time.time()
        model.l_time = 0
        model.new_tokens = 0
        import pdb;pdb.set_trace()
        generated_ids = model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print('New tokens', model.new_tokens)
        print(f'Language model duration {model.l_time}')
        print(f'E2E duration {time.time() - e2e_start} s')
    print(output_text)

if __name__ == '__main__':
    torch_torch()