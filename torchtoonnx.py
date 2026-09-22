import torch    
import importlib
from lib.models.mixformer2_vit import build_mixformer2_vit



def get_parameters(tracker_params=None):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format("mixformer2_vit"))
    search_area_scale = None
    if tracker_params is not None and 'search_area_scale' in tracker_params:
        search_area_scale = tracker_params['search_area_scale']
    model = ''
    if tracker_params is not None and 'model' in tracker_params:
        model = tracker_params['model']
    params = param_module.parameters("student_288_depth12", model, search_area_scale)
    if tracker_params is not None:
        for param_k, v in tracker_params.items():
            setattr(params, param_k, v)
    return params

params = get_parameters()
network = build_mixformer2_vit(params.cfg)

checkpoint = torch.load(    
    "/kaggle/input/models/huynhat15/mixformerv2/pytorch/ep0010/1/MixFormer_ep0010.pth.tar",
    map_location='cpu',
    weights_only=False
)

network.load_state_dict(
    checkpoint['net'],
    strict=True
)

network.eval()

# 1. Sửa lỗi đánh máy phần tạo dữ liệu giả (Dummy Inputs)
# Đảm bảo chúng cùng kiểu dữ liệu (float) và trên cùng thiết bị (CPU/GPU) với model
template_input = torch.randn(1, 3, 128, 128).float()
onlinetemplate_input = torch.randn(1, 3, 128, 128).float()
search_input = torch.randn(1, 3, 288, 288).float()


class MixFormerONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, template, online_template, search):
        # Truyền tĩnh softmax=True (hoặc False tùy cấu hình của bạn)
        # Các tham số khác như run_score_head để default
        pred_boxes = self.model(
            template=template, 
            online_template=online_template, 
            search=search, 
            softmax=True, 
            run_score_head=True
        )
        return pred_boxes

# Bọc mô hình
onnx_model = MixFormerONNXWrapper(network)
onnx_model.eval()

onnx_file_path = "mixformer2_vit.onnx"

print("Đang tiến hành export sang ONNX...")
torch.onnx.export(
    onnx_model,                                     # Mô hình đã được bọc
    (template_input, onlinetemplate_input, search_input), # Tuple chứa các input tensors
    onnx_file_path,                                 # Đường dẫn lưu file
    export_params=True,                             # Lưu trọng số vào trong file ONNX
    opset_versio=10,                               # Opset version (11 hoặc 12, 13 phù hợp cho Transformer)
    do_constant_folding=True,                       # Tối ưu hóa (tính trước các hằng số)
    input_names=['template', 'online_template', 'search'], # Đặt tên cho các input node
    output_names=['pred_boxes'],                    # Đặt tên cho output node
    
)

print(f"Export thành công! File lưu tại: {onnx_file_path}")