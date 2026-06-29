# InsightFace 人脸识别 Demo

基于 InsightFace 的人物身份识别和停留时间记录系统，支持自定义人脸库和 GPU 加速。

## 功能特性

- 实时人脸识别
- 自定义人脸库，支持录入具体人物信息
- 自动记录每个人物在摄像头中停留的时间
- 人脸框标注和信息显示（已注册的人脸显示绿色框，未知人脸显示橙色框）
- 实时显示当前检测到的人物列表和停留时长
- 支持人脸库持久化保存和加载
- **支持 NVIDIA GPU 加速（RTX 3050 等）**

## 环境要求

### GPU 加速要求（可选但推荐）
- NVIDIA 显卡（RTX 3050 或其他支持 CUDA 的显卡）
- CUDA Toolkit 11.x 或 12.x
- 最新的 NVIDIA 显卡驱动

### CPU 模式
- 无需特殊配置，直接运行即可

## 安装依赖

### 第一步：检查 GPU 环境（推荐）

```bash
python check_gpu.py
```

这将检查你的系统是否支持 CUDA GPU 加速。

### 第二步：安装依赖

```bash
pip install -r requirements.txt
```

**注意**：`requirements.txt` 中默认使用 `onnxruntime-gpu`。如果你的系统不支持 GPU，可以修改为 `onnxruntime`。

## 使用方法

### GPU/CPU 模式切换

程序默认使用 GPU 模式。如需强制使用 CPU，添加 `--cpu` 参数：

```bash
# GPU 模式（默认）
python face_recognition_demo.py

# CPU 模式
python face_recognition_demo.py --cpu

# 注册工具也支持同样的参数
python register_face.py --cpu
```

### 方式一：使用人脸注册工具（推荐）

```bash
python register_face.py
```

注册工具提供以下功能：
1. **从图片注册** - 使用单张图片注册人脸
2. **从摄像头注册** - 实时从摄像头采集人脸并注册
3. **批量从目录注册** - 批量从 `faces` 文件夹注册（文件名作为人名）
4. **查看人脸库** - 查看当前已注册的所有人
5. **清空人脸库** - 清除所有人脸数据

### 方式二：直接运行识别程序

```bash
python face_recognition_demo.py
```

运行时快捷键：
- `q` - 退出程序
- `r` - 注册当前检测到的第一张人脸（需要在终端输入人名）

### 批量注册方式

将人物照片放入 `faces` 文件夹，文件名为人物姓名（如：张三.jpg），然后运行：

```bash
python face_recognition_demo.py --load-dir
```

或使用注册工具的选项3。

## 性能优化建议

### 解决卡顿问题

程序已经做了多重优化，你可以根据情况调整：

#### 1. 跳帧检测（推荐）
使用 `--skip=N` 参数调整检测频率：
```bash
# 默认：每3帧检测一次（速度提升约3倍）
python face_recognition_demo.py

# 更快：每5帧检测一次（速度提升约5倍）
python face_recognition_demo.py --skip=4

# 最快：每10帧检测一次
python face_recognition_demo.py --skip=9
```

#### 2. 使用 CPU 模式（如果 GPU 不可用）
```bash
python face_recognition_demo.py --cpu
```

#### 3. 已应用的优化
- ✅ 降低检测分辨率：从 640x640 到 256x256
- ✅ 降低摄像头分辨率：到 640x480
- ✅ 跳帧检测：默认每3帧检测一次
- ✅ 人脸缓存：使用上一帧检测结果

### 性能对比

| 模式 | 跳帧数 | 相对速度 | 推荐场景 |
|------|--------|----------|----------|
| 最高精度 | 0 | 1x | 追求精度 |
| **平衡（默认）** | **2** | **3x** | **日常使用** |
| 快速 | 4 | 5x | 稍许卡顿 |
| 最快 | 9 | 10x | 严重卡顿 |

## 人脸库说明

- 人脸库数据保存在 `face_database.pkl` 文件中
- 已注册的人脸会显示真实姓名（绿色框）
- 未注册的人脸会显示 Person_0, Person_1 等ID（橙色框）
- 注册的人脸会自动保存，下次启动时自动加载

## 系统工作原理

- 使用 InsightFace 进行人脸检测和特征提取
- 通过人脸特征向量比对进行身份识别
- 记录每个人物进入画面的时间
- 如果人物离开画面超过2秒，重新进入时会重新计时
- 停留时间显示在人脸框上方

## 常见问题

### Q: 提示 CUDA 不可用怎么办？
A: 运行 `python check_gpu.py` 检查环境，或使用 `--cpu` 参数切换到 CPU 模式。

### Q: GPU 模式还是有点卡？
A: 确保没有其他程序占用 GPU，检查任务管理器中的 GPU 使用率。

## 注意事项

- 首次运行时会自动下载 InsightFace 模型文件
- 需要连接网络下载模型（约200MB）
- 确保摄像头设备可用
- 注册人脸时建议使用清晰的正面照片
- GPU 模式需要正确安装 CUDA 环境和 onnxruntime-gpu


