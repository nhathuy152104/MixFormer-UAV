import sys
import numpy as np
from rknn.api import RKNN

DATASET_PATH = '/home/arsene_lupin/HuyWorkspace/VtWork/OSTrack-Rockchip/path.txt'
DEFAULT_RKNN_PATH = 'ostrack.rknn'
DEFAULT_QUANT = True # Khuyến nghị test FP16 trước với ViT

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path platform [dtype(optional)] [output_rknn_path]".format(sys.argv[0]))
        exit(1)
    
    model_path = sys.argv[1]
    platform = sys.argv[2]
    
    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False
            
    output_path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_RKNN_PATH
    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    rknn = RKNN(verbose=True) # Bật verbose để xem log compile của từng layer

    print('--> Config model')
    # SỬA Ở ĐÂY: OSTrack có 2 input (template, search) và dùng chuẩn hóa ImageNet
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53], [123.675, 116.28, 103.53]],
        std_values=[[58.395, 57.12, 57.375], [58.395, 57.12, 57.375]],
        target_platform=platform,
        optimization_level=3, # Bắt buộc bật mức 3 để tối ưu ViT
        quantized_algorithm='normal' 
    )
    print('done')

    print('--> Loading model')
    if rknn.load_onnx(model=model_path) != 0:
        print('Load model failed!')
        exit(1)
    print('done')

    print('--> Building model')
    # Lưu ý: Với ViT, nếu bật do_quant=True (INT8), compiler có thể bối rối
    # và đẩy rất nhiều node về CPU. Hãy test với FP16 (do_quant=False) trước.
    if rknn.build(do_quantization=do_quant, dataset=DATASET_PATH) != 0:
        print('Build model failed!')
        exit(1)
    print('done')

    print('--> Export rknn model')
    if rknn.export_rknn(output_path) != 0:
        print('Export rknn model failed!')
        exit(1)
    print('done')

    # ==========================================
    # PHẦN MỚI: TÌM NGUYÊN NHÂN 1 FPS (PROFILING)
    # Cắm board RK3576 vào máy tính (hoặc chạy trực tiếp trên board) để test
    # ==========================================
    print('--> Init runtime environment & Profiling (Requires connected RK3576 board)')
    ret = rknn.init_runtime(target=platform, perf_debug=True)
    if ret == 0:
        # Giả lập input ngẫu nhiên theo shape của OSTrack (template 128x128, search 256x256)
        # Sửa lại shape này cho đúng với model ONNX thực tế của bạn
        dummy_template = np.random.randint(0, 255, size=(1, 3, 128, 128), dtype=np.uint8)
        dummy_search = np.random.randint(0, 255, size=(1, 3, 256, 256), dtype=np.uint8)
        
        print('--> Running eval_perf')
        # Lệnh này sẽ in ra bảng thời gian chạy của TỪNG LAYER và cho biết layer nào chạy trên CPU/NPU
        rknn.eval_perf(is_print=True)
    else:
        print('Skip profiling because init_runtime failed (Board not connected?).')

    rknn.release()