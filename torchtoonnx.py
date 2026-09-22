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
    strict=False
)

network.eval()

network = network.cpu()

# Đảm bảo self.indice trong lớp Head nằm trên CPU (sửa lỗi lệch device)
for m in network.modules():
    if hasattr(m, 'indice') and isinstance(m.indice, torch.Tensor):
        m.indice = m.indice.cpu()

# 2. Tạo Dummy Inputs trên CPU
template_input = torch.randn(1, 3, 128, 128, device='cpu').float()
onlinetemplate_input = torch.randn(1, 3, 128, 128, device='cpu').float()
search_input = torch.randn(1, 3, 288, 288, device='cpu').float()

# 3. Tạo Wrapper để giấu các biến boolean (softmax, run_score_head) khỏi ONNX Tracer
class MixFormerONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, template, online_template, search):
        # Trả về trực tiếp pred_boxes (có thể tuỳ chỉnh tùy vào hàm forward của bạn)
        return self.model(
            template=template, 
            online_template=online_template, 
            search=search, 
            softmax=True, 
            run_score_head=True
        )

onnx_model = MixFormerONNXWrapper(network)
onnx_model.eval()

onnx_file_path = "mixformer2_vit_opset10.onnx"
print("Đang tiến hành export sang ONNX (Opset 10)...")

# 4. Export sang ONNX với Opset 10
with torch.no_grad():
    torch.onnx.export(
        onnx_model,
        (template_input, onlinetemplate_input, search_input),
        onnx_file_path,
        export_params=True,
        opset_version=10,           # <--- CHỈ ĐỊNH PHIÊN BẢN 10
        do_constant_folding=True,
        input_names=['template', 'online_template', 'search'],
        output_names=['pred_boxes'],
        dynamic_axes={
            'template': {0: 'batch_size'},
            'online_template': {0: 'batch_size'},
            'search': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'}
        }
    )

print(f"Export thành công: {onnx_file_path}")