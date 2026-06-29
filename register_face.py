import cv2
import os
import sys
from face_recognition_demo import FaceRecognitionDemo

def get_demo(use_gpu=True):
    return FaceRecognitionDemo(use_gpu=use_gpu)

def register_from_image(use_gpu=True):
    print("=== 从图片注册人脸 ===")
    print("\n说明:")
    print("- 单张图片: 直接注册")
    print("- 多张图片: 同一个人名的多张照片会被添加为不同角度")
    
    image_path = input("请输入图片路径: ").strip()
    name = input("请输入人名: ").strip()
    
    if not os.path.exists(image_path):
        print("图片不存在!")
        return
    
    demo = get_demo(use_gpu)
    demo.load_face_from_image(image_path, name)

def register_from_webcam_single(use_gpu=True):
    print("=== 从摄像头注册单个人脸 ===")
    name = input("请输入人名: ").strip()
    
    demo = get_demo(use_gpu)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 's' 保存当前画面并注册，按 'q' 取消")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = demo.app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Register Face', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("取消注册")
            break
        elif key == ord('s'):
            if len(faces) > 0:
                demo.register_face(name, faces[0].embedding, add_to_existing=False)
                print(f"成功注册: {name}")
                break
            else:
                print("未检测到人脸，请重试")
    
    cap.release()
    cv2.destroyAllWindows()

def register_from_webcam_multi(use_gpu=True):
    print("=== 多角度人脸注册 ===")
    print("=" * 60)
    print("我们将依次采集5个角度的人脸:")
    print("  1. 正脸")
    print("  2. 向左转头 (约30度)")
    print("  3. 向右转头 (约30度)")
    print("  4. 微微抬头")
    print("  5. 微微低头")
    print("=" * 60)
    
    name = input("请输入人名: ").strip()
    
    demo = get_demo(use_gpu)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    angles = [
        "正脸",
        "向左转头 (约30度)",
        "向右转头 (约30度)",
        "微微抬头",
        "微微低头"
    ]
    
    current_angle = 0
    registered_count = 0
    
    while current_angle < len(angles):
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = demo.app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # 显示提示
        cv2.putText(frame, f"角度 {current_angle + 1}/{len(angles)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, angles[current_angle], (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        cv2.putText(frame, "按 's' 保存，'q' 跳过，'ESC' 退出", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Multi-Angle Registration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("退出注册")
            break
        elif key == ord('q'):
            print(f"跳过角度: {angles[current_angle]}")
            current_angle += 1
        elif key == ord('s'):
            if len(faces) > 0:
                if registered_count == 0:
                    demo.register_face(name, faces[0].embedding, add_to_existing=False)
                else:
                    demo.register_face(name, faces[0].embedding, add_to_existing=True)
                registered_count += 1
                print(f"✓ 已保存: {angles[current_angle]} ({registered_count}/{len(angles)})")
                current_angle += 1
            else:
                print("未检测到人脸，请重试")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if registered_count > 0:
        print(f"\n✓ 注册完成！共保存 {registered_count} 个角度")
    else:
        print("\n未保存任何角度")

def register_from_directory(use_gpu=True):
    print("=== 批量从目录注册人脸 ===")
    print("将从 'faces' 文件夹中读取所有图片，文件名作为人名")
    print("\n提示:")
    print("- 同一人多张照片: 命名为 '张三_1.jpg', '张三_2.jpg'")
    print("  或者都叫 '张三.jpg' 但放在不同文件夹")
    
    demo = get_demo(use_gpu)
    demo.load_faces_from_directory()
    print(f"\n注册完成，当前人脸库共 {len(demo.registered_faces)} 人")

def list_database(use_gpu=True):
    print("=== 人脸库列表 ===")
    demo = get_demo(use_gpu)
    if len(demo.registered_faces) == 0:
        print("人脸库为空")
    else:
        print(f"\n共 {len(demo.registered_faces)} 人:\n")
        for name, data in demo.registered_faces.items():
            count = len(data['embeddings'])
            print(f"  • {name} ({count} 个角度)")

def clear_database(use_gpu=True):
    print("=== 清空人脸库 ===")
    confirm = input("确定要清空所有人脸数据吗？(yes/no): ").strip().lower()
    if confirm == 'yes':
        demo = get_demo(use_gpu)
        demo.registered_faces = {}
        demo.save_face_database()
        print("人脸库已清空")
    else:
        print("操作取消")

def main():
    use_gpu = '--cpu' not in sys.argv
    
    while True:
        print("\n" + "=" * 60)
        print("人脸注册系统")
        print("=" * 60)
        print(f"当前模式: {'GPU' if use_gpu else 'CPU'}")
        print("\n注册选项:")
        print("  1. 单张图片注册")
        print("  2. 摄像头单次注册")
        print("  3. 摄像头多角度注册 (推荐!)")
        print("  4. 批量从目录注册")
        print("\n管理选项:")
        print("  5. 查看人脸库")
        print("  6. 清空人脸库")
        print("  7. 退出")
        print("=" * 60)
        
        choice = input("\n请选择操作 (1-7): ").strip()
        
        if choice == '1':
            register_from_image(use_gpu)
        elif choice == '2':
            register_from_webcam_single(use_gpu)
        elif choice == '3':
            register_from_webcam_multi(use_gpu)
        elif choice == '4':
            register_from_directory(use_gpu)
        elif choice == '5':
            list_database(use_gpu)
        elif choice == '6':
            clear_database(use_gpu)
        elif choice == '7':
            print("再见!")
            break
        else:
            print("无效选择，请重试")

if __name__ == "__main__":
    main()
