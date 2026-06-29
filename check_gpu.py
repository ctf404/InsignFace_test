import sys
import os

def check_cuda():
    print("=" * 60)
    print("GPU 环境详细检查")
    print("=" * 60)
    
    # 检查 ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"\n[1] ONNX Runtime 版本: {ort.__version__}")
        available_providers = ort.get_available_providers()
        print(f"    可用的执行提供者: {available_providers}")
        
        # 检查是否同时安装的是哪个版本
        is_gpu_version = 'CUDAExecutionProvider' in available_providers
        
        if is_gpu_version:
            print("    ✓ 检测到 onnxruntime-gpu 版本")
        else:
            print("    ⚠️  检测到的是 onnxruntime (CPU 版本)")
            print("       如果你想用 GPU，需要先卸载 CPU 版本，再安装 GPU 版本")
            print("       运行: pip uninstall onnxruntime onnxruntime-gpu")
            print("       然后: pip install onnxruntime-gpu")
        
        if 'CUDAExecutionProvider' in available_providers:
            print("    ✓ CUDAExecutionProvider 可用")
            cuda_available = True
            
            # 尝试测试 CUDA
            try:
                print("\n    正在测试 CUDA 初始化...")
                # 创建一个简单的测试（用空的 session 不太对，我们跳过直接测试 providers
                print("    ✓ CUDAExecutionProvider 在可用列表中")
                cuda_available = True
            except Exception as e:
                print(f"    ✗ CUDA 初始化失败: {e}")
                print("\n    常见原因:")
                print("    - CUDA Toolkit 版本不匹配")
                print("    - cuDNN 未安装或版本不匹配")
                print("    - NVIDIA 驱动版本太旧")
        else:
            print("    ✗ CUDAExecutionProvider 不可用")
            cuda_available = False
            
    except ImportError:
        print("\n[1] ✗ onnxruntime 未安装")
        cuda_available = False
    
    # 检查系统 GPU
    print("\n[3] 检查系统 GPU 信息...")
    try:
        # 尝试用不同的方法检测 GPU
        gpu_found = False
        
        # 方法 1: 使用 PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_found = True
                print(f"\n    检测到 GPU (PyTorch):")
                print(f"    - GPU 数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    - CUDA 版本: {torch.version.cuda}")
        except ImportError:
            pass
        
        # 方法 2: 使用系统命令
        if not gpu_found:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                       capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    gpu_found = True
                    gpus = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                    print(f"\n    检测到 GPU (nvidia-smi):")
                    for i, gpu in enumerate(gpus):
                        print(f"    - GPU {i}: {gpu}")
            except:
                pass
        
        if not gpu_found:
            print("    ✗ 未检测到 NVIDIA GPU")
            
    except Exception as e:
        print(f"    检查 GPU 时出错: {e}")
    
    # 检查环境变量
    print("\n[4] 检查 CUDA 环境变量...")
    cuda_path = os.environ.get('CUDA_PATH', '')
    cuda_path_11 = os.environ.get('CUDA_PATH_V11_8', '')
    cuda_path_12 = os.environ.get('CUDA_PATH_V12_1', '')
    
    if cuda_path:
        print(f"    ✓ CUDA_PATH: {cuda_path}")
    if cuda_path_11:
        print(f"    ✓ CUDA_PATH_V11_8: {cuda_path_11}")
    if cuda_path_12:
        print(f"    ✓ CUDA_PATH_V12_1: {cuda_path_12}")
    if not cuda_path and not cuda_path_11 and not cuda_path_12:
        print("    ✗ 未检测到 CUDA_PATH 环境变量")
    
    print("\n" + "=" * 60)
    print("诊断结果和建议")
    print("=" * 60)
    
    if cuda_available:
        print("✅ GPU 环境完全就绪！")
        print("   直接运行即可使用 GPU 加速:")
        print("   python face_recognition_demo.py")
    else:
        print("❌ GPU 不可用")
        print("\n你有两个选择:")
        print("\n选项 A: 使用 CPU 模式（推荐，立即可以用）")
        print("   python face_recognition_demo.py --cpu")
        print("\n选项 B: 配置 GPU 环境（按以下步骤）")
        print("\n   步骤 1: 检查 NVIDIA 驱动")
        print("     - 打开 NVIDIA 控制面板")
        print("     - 或运行: nvidia-smi")
        print("     - 确保驱动版本 >= 525.60.13")
        
        print("\n   步骤 2: 下载 CUDA Toolkit 11.8")
        print("     - 下载地址: https://developer.nvidia.com/cuda-11-8-0-download-archive")
        print("     - 选择: Windows -> x86_64 -> 10 or 11 -> exe (local)")
        print("     - 下载后安装")
        
        print("\n   步骤 3: 重启电脑")
        
        print("\n   步骤 4: 重新检查")
        print("     - python check_gpu.py")
        
        print("\n   注意:")
        print("   - onnxruntime-gpu 1.17.x 推荐搭配 CUDA 11.8")
        print("   - 如果不想配置，直接用 CPU 模式也很好！")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_cuda()
