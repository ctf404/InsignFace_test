import subprocess
import sys

print("=" * 60)
print("GPU 环境快速修复工具")
print("=" * 60)

# 步骤 1: 检查当前安装的包
print("\n[1] 检查当前安装的 onnxruntime 包...")
try:
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                          capture_output=True, text=True)
    lines = result.stdout.lower().split('\n')
    
    has_cpu = any('onnxruntime' in line and not 'onnxruntime-gpu' in line for line in lines)
    has_gpu = any('onnxruntime-gpu' in line for line in lines)
    
    print(f"    onnxruntime (CPU版): {'✓ 已安装' if has_cpu else '✗ 未安装'}")
    print(f"    onnxruntime-gpu (GPU版): {'✓ 已安装' if has_gpu else '✗ 未安装'}")
    
    if has_cpu and has_gpu:
        print("\n⚠️  检测到同时安装了 CPU 和 GPU 版本！")
        print("   这会导致冲突，GPU 无法使用")
        
        confirm = input("\n是否需要修复？(y/n): ").strip().lower()
        if confirm == 'y':
            print("\n正在卸载两个版本...")
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'onnxruntime', 'onnxruntime-gpu'])
            
            print("\n正在安装 GPU 版本...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnxruntime-gpu'])
            
            print("\n✓ 修复完成！")
            print("现在请运行: python check_gpu.py")
    elif not has_gpu:
        print("\n⚠️  未检测到 onnxruntime-gpu")
        confirm = input("\n是否安装 onnxruntime-gpu？(y/n): ").strip().lower()
        if confirm == 'y':
            print("\n正在安装 onnxruntime-gpu...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnxruntime-gpu'])
            print("\n✓ 安装完成！")
    else:
        print("\n✓ onnxruntime-gpu 已安装，看起来没问题")
        
except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 60)
