import torch
import onnx
import onnxsim
from onnxmltools.utils import float16_converter


model_path = 'weights/v8_n_pose.pt'
device = 'cpu'
onnx_path = 'weights/v8_n_pose.onnx'
simplified_path = 'weights/v8_n_pose_simplified.onnx'

# 加载模型
with torch.no_grad():
    from torch.cuda.amp import autocast

    with autocast():
        model = torch.load(model_path, map_location=device)['model'].float().fuse().eval()

    stride = int(model.stride.max())

    # 转换模型为ONNX格式
    input_shape = (1, 3, 512, 512)
    torch.onnx.export(model, torch.randn(*input_shape, device=device), onnx_path,
                      input_names=['images'], output_names=['output'], export_params=True,
                      opset_version=11, do_constant_folding=True)

# 加载原始的ONNX模型
original_model = onnx.load(onnx_path)

# 使用onnxsim进行模型简化
simplified_model, _ = onnxsim.simplify(original_model)
# 保存简化后的模型为新的ONNX文件
onnx.save(simplified_model, simplified_path)

