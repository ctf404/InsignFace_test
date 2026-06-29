import cv2
import numpy as np
import time
import os
import pickle
from insightface.app import FaceAnalysis
from collections import defaultdict

class FaceRecognitionDemo:
    def __init__(self, faces_dir="faces", database_file="face_database.pkl", use_gpu=True, det_size=(256, 256), skip_frames=2):
        print("=" * 60)
        print("初始化人脸识别系统...")
        print("=" * 60)
        
        # 检查 CUDA 是否可用
        cuda_available = False
        try:
            import onnxruntime as ort
            print(f"\n[1] ONNX Runtime 版本: {ort.__version__}")
            available_providers = ort.get_available_providers()
            print(f"    可用的执行提供者: {available_providers}")
            
            cuda_available = 'CUDAExecutionProvider' in available_providers
            
            if cuda_available:
                print("    ✓ CUDAExecutionProvider 可用")
            else:
                print("    ✗ CUDAExecutionProvider 不可用")
        except Exception as e:
            print(f"    检查 ONNX Runtime 时出错: {e}")
        
        # 根据可用性选择运行模式
        if use_gpu and cuda_available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
            print("\n✅ 将使用 GPU (CUDA) 进行推理")
        else:
            if use_gpu and not cuda_available:
                print("\n⚠️  CUDA 不可用，自动降级到 CPU 模式")
                print("   原因: 未检测到 CUDAExecutionProvider")
                print("   解决: 运行 python check_gpu.py 查看详细信息")
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            print("\n💻 将使用 CPU 进行推理")
        
        print("\n正在加载模型...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=".",
            providers=providers
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print("✓ 模型加载完成")
        print("=" * 60)
        self.registered_faces = {}  # 已注册的人脸（保存到数据库）
        # 结构: { '人名': { 'embeddings': [emb1, emb2, ...], 'timestamps': [t1, t2, ...] } }
        self.temp_faces = {}  # 临时识别的人脸（不保存）
        self.face_timers = defaultdict(lambda: {'enter_time': None, 'total_time': 0, 'last_seen': None})
        self.face_counter = 0
        self.threshold = 0.6
        self.faces_dir = faces_dir
        self.database_file = database_file
        
        # 多帧融合相关
        self.frame_history = []  # 存储最近几帧的检测结果
        self.max_history = 10  # 保留最近10帧
        self.load_face_database()
        
        # 性能优化参数
        self.skip_frames = skip_frames  # 跳帧检测
        self.frame_count = 0
        self.last_faces = []  # 缓存上一帧的人脸
        
    def register_face(self, name, face_embedding, add_to_existing=True):
        """注册人脸，支持添加多个角度的embedding"""
        import time
        current_time = time.time()
        
        if name not in self.registered_faces:
            self.registered_faces[name] = {
                'embeddings': [face_embedding],
                'timestamps': [current_time],
                'angles': ['正面']
            }
            print(f"✓ 首次注册: {name}")
        else:
            if add_to_existing:
                self.registered_faces[name]['embeddings'].append(face_embedding)
                self.registered_faces[name]['timestamps'].append(current_time)
                self.registered_faces[name]['angles'].append(f'角度{len(self.registered_faces[name]["angles"])+1}')
                print(f"✓ 添加新角度: {name} (共{len(self.registered_faces[name]['embeddings'])}个角度)")
            else:
                self.registered_faces[name] = {
                    'embeddings': [face_embedding],
                    'timestamps': [current_time],
                    'angles': ['正面']
                }
                print(f"✓ 覆盖注册: {name}")
        
        self.save_face_database()
    
    def load_face_database(self):
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'rb') as f:
                    data = pickle.load(f)
                
                # 兼容旧版本格式
                if isinstance(data, dict):
                    self.registered_faces = {}
                    for name, value in data.items():
                        if isinstance(value, dict) and 'embeddings' in value:
                            self.registered_faces[name] = value
                        else:
                            # 旧版本格式，转换
                            self.registered_faces[name] = {
                                'embeddings': [value],
                                'timestamps': [0],
                                'angles': ['旧版']
                            }
                
                print(f"已加载人脸库，共 {len(self.registered_faces)} 人")
                for name in self.registered_faces:
                    count = len(self.registered_faces[name]['embeddings'])
                    print(f"  - {name}: {count} 个角度")
            except Exception as e:
                print(f"加载人脸库失败: {e}")
                self.registered_faces = {}
        else:
            print("人脸库不存在，将创建新的人脸库")
    
    def save_face_database(self):
        try:
            with open(self.database_file, 'wb') as f:
                pickle.dump(self.registered_faces, f)
        except Exception as e:
            print(f"保存人脸库失败: {e}")
    
    def load_face_from_image(self, image_path, name):
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return False
        
        faces = self.app.get(img)
        if len(faces) == 0:
            print(f"图片中未检测到人脸: {image_path}")
            return False
        
        if len(faces) > 1:
            print(f"警告: 图片中检测到多张人脸，将使用第一张: {image_path}")
        
        self.register_face(name, faces[0].embedding)
        print(f"成功注册: {name}")
        return True
    
    def load_faces_from_directory(self):
        if not os.path.exists(self.faces_dir):
            print(f"人脸图片目录不存在: {self.faces_dir}")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for filename in os.listdir(self.faces_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.faces_dir, filename)
                self.load_face_from_image(image_path, name)
        
    def compare_faces(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def get_face_name(self, embedding):
        max_similarity = 0
        best_match = None
        
        # 先检查已注册的人脸（支持多embedding）
        for name, face_data in self.registered_faces.items():
            embeddings = face_data['embeddings']
            
            # 计算与所有已存embedding的相似度，取最高的
            for known_emb in embeddings:
                similarity = self.compare_faces(embedding, known_emb)
                if similarity > max_similarity and similarity > self.threshold:
                    max_similarity = similarity
                    best_match = name
        
        # 如果已注册人脸中没找到，再检查临时人脸
        if best_match is None:
            for name, temp_embedding in self.temp_faces.items():
                similarity = self.compare_faces(embedding, temp_embedding)
                if similarity > max_similarity and similarity > self.threshold:
                    max_similarity = similarity
                    best_match = name
        
        # 如果都没找到，创建临时 ID（不保存到数据库）
        if best_match is None:
            best_match = f"Person_{self.face_counter}"
            self.temp_faces[best_match] = embedding
            self.face_counter += 1
        
        return best_match
    
    def update_frame_history(self, frame_detections):
        """更新帧历史，用于多帧融合"""
        self.frame_history.append(frame_detections)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
    
    def get_fused_prediction(self, current_embedding):
        """使用多帧融合进行更稳定的识别"""
        if len(self.frame_history) < 3:
            # 历史数据不够，直接用当前帧
            return self.get_face_name(current_embedding)
        
        # 收集最近几帧的识别结果
        name_votes = {}
        
        # 当前帧的识别
        current_name = self.get_face_name(current_embedding)
        name_votes[current_name] = name_votes.get(current_name, 0) + 2  # 当前帧权重更高
        
        # 历史帧的识别结果
        for frame_data in self.frame_history[-5:]:  # 只看最近5帧
            for face_emb, name in frame_data:
                name_votes[name] = name_votes.get(name, 0) + 1
        
        # 找出得票最高的
        if name_votes:
            sorted_names = sorted(name_votes.items(), key=lambda x: x[1], reverse=True)
            return sorted_names[0][0]
        
        return current_name
    
    def update_face_timer(self, face_id):
        current_time = time.time()
        
        if self.face_timers[face_id]['enter_time'] is None:
            self.face_timers[face_id]['enter_time'] = current_time
            self.face_timers[face_id]['last_seen'] = current_time
        else:
            if current_time - self.face_timers[face_id]['last_seen'] > 2.0:
                self.face_timers[face_id]['enter_time'] = current_time
            
            self.face_timers[face_id]['last_seen'] = current_time
            self.face_timers[face_id]['total_time'] = current_time - self.face_timers[face_id]['enter_time']
    
    def get_stay_time(self, face_id):
        return self.face_timers[face_id]['total_time']
    
    def run(self, load_from_dir=False):
        if load_from_dir:
            self.load_faces_from_directory()
        
        cap = cv2.VideoCapture(0)
        
        # 降低摄像头分辨率以提高性能
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("\n摄像头已打开")
        print("-" * 60)
        print("快捷键说明:")
        print("  q   - 退出程序")
        print("  r   - 注册当前检测到的第一张人脸（新用户）")
        print("  a   - 添加新角度到已存在的用户")
        print("  v   - 切换视角提示显示")
        print(f"性能模式: 每{self.skip_frames + 1}帧检测一次人脸")
        print("-" * 60)
        
        show_angle_guide = True
        current_registration_name = None
        registration_step = 0
        
        # 推荐的角度列表
        angles = [
            "正脸",
            "向左转头 (约30度)",
            "向右转头 (约30度)",
            "微微抬头",
            "微微低头"
        ]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break
            
            # 跳帧检测优化
            if self.frame_count % (self.skip_frames + 1) == 0:
                # 这一帧做人脸检测
                faces = self.app.get(frame)
                self.last_faces = faces
                
                # 更新帧历史（用于多帧融合）
                frame_detections = []
                for face in faces:
                    embedding = face.embedding
                    face_id = self.get_face_name(embedding)
                    frame_detections.append((embedding, face_id))
                    self.update_face_timer(face_id)
                
                self.update_frame_history(frame_detections)
            else:
                # 使用上一帧的检测结果
                faces = self.last_faces
                
                # 更新最后看到的时间
                for face in faces:
                    embedding = face.embedding
                    face_id = self.get_face_name(embedding)
                    self.face_timers[face_id]['last_seen'] = time.time()
            
            # 绘制人脸框（每一帧都绘制）
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                
                # 使用多帧融合进行识别
                face_id = self.get_fused_prediction(embedding)
                stay_time = self.get_stay_time(face_id)
                
                color = (0, 255, 0) if not face_id.startswith('Person_') else (0, 165, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                label = f"{face_id} | {stay_time:.1f}s"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(frame, (bbox[0], bbox[1] - 10 - label_size[1]), 
                            (bbox[0] + label_size[0], bbox[1] - 5), color, -1)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 显示角度提示
            if show_angle_guide:
                y_guide = 50
                cv2.putText(frame, "建议注册角度:", (10, y_guide), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
                for i, angle in enumerate(angles):
                    y_guide += 25
                    color = (100, 255, 100)
                    cv2.putText(frame, f"{i+1}. {angle}", (25, y_guide), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
            # 显示侧边栏
            y_offset = 30
            cv2.putText(frame, "Detected Persons:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            for face_id in self.face_timers:
                if self.face_timers[face_id]['last_seen'] is not None and \
                   time.time() - self.face_timers[face_id]['last_seen'] < 5.0:
                    y_offset += 25
                    stay_time = self.face_timers[face_id]['total_time']
                    cv2.putText(frame, f"{face_id}: {stay_time:.1f}s", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition Demo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # 注册新人
                if self.frame_count % (self.skip_frames + 1) != 0:
                    faces = self.app.get(frame)
                if len(faces) > 0:
                    name = input("请输入要注册的人名: ").strip()
                    if name:
                        self.register_face(name, faces[0].embedding, add_to_existing=False)
                        print(f"\n提示: 可以按 'a' 继续添加其他角度")
                else:
                    print("未检测到人脸，无法注册")
            elif key == ord('a'):
                # 添加新角度
                if self.frame_count % (self.skip_frames + 1) != 0:
                    faces = self.app.get(frame)
                if len(faces) > 0:
                    print("\n已注册的人脸:")
                    for name in self.registered_faces:
                        count = len(self.registered_faces[name]['embeddings'])
                        print(f"  - {name} ({count} 个角度)")
                    name = input("请输入要添加角度的人名: ").strip()
                    if name in self.registered_faces:
                        self.register_face(name, faces[0].embedding, add_to_existing=True)
                    else:
                        print(f"未找到 '{name}'，请先使用 'r' 注册")
                else:
                    print("未检测到人脸，无法添加角度")
            elif key == ord('v'):
                show_angle_guide = not show_angle_guide
                print(f"视角提示: {'显示' if show_angle_guide else '隐藏'}")
            
            self.frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    use_gpu = '--cpu' not in sys.argv
    load_from_dir = '--load-dir' in sys.argv
    
    # 性能参数
    skip_frames = 2  # 默认每3帧检测一次
    
    # 解析命令行参数
    for arg in sys.argv:
        if arg.startswith('--skip='):
            try:
                skip_frames = int(arg.split('=')[1])
            except:
                pass
    
    print("正在初始化人脸识别系统...")
    demo = FaceRecognitionDemo(
        use_gpu=use_gpu,
        det_size=(256, 256),
        skip_frames=skip_frames
    )
    demo.run(load_from_dir=load_from_dir)
