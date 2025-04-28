import onnx
import onnxruntime as ort
import numpy as np
import torch

# 加载 ONNX 模型
model_path = "./tmp/trt_engines/Qwen2_5-VL/fp16/1-gpu/vision/onnx/model.onnx"

image = torch.load('image.pt').numpy()
rotary_pos_emb = torch.load('rotary_pos_emb.pt').numpy()
full_attention_mask = torch.load('full_attention_mask.pt').numpy()
window_attention_mask = torch.load('window_attention_mask.pt').numpy()
# window_index = torch.load('window_index.pt').numpy()

# 使用 onnxruntime 创建推理会话
session = ort.InferenceSession(model_path)

# 获取模型的输入信息（名称、形状等）
for i in range(4):
    input_name = session.get_inputs()[i].name
    input_shape = session.get_inputs()[i].shape
    input_type = session.get_inputs()[i].type
    print(f"Input Name: {input_name}")
    print(f"Input Shape: {input_shape}")
    print(f"Input Type: {input_type}")


# 运行推理
outputs = session.run(None, {'input': image,
                             'rotary_pos_emb': rotary_pos_emb,
                             'full_attention_mask': full_attention_mask,
                             'window_attention_mask': window_attention_mask,
                            #  'window_index': window_index,
                             })

# 获取输出信息（名称、形状等）
output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape
print(f"Output Name: {output_name}")
print(f"Output Shape: {output_shape}")

# 处理输出（这里只是打印输出）
print(f"Output: {outputs[0]}")
image_embeds = torch.tensor(outputs[0])
import  pdb; pdb.set_trace()
torch.save(image_embeds, 'image_embeds_from_onnx.pt')