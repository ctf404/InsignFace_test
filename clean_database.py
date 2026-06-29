import os
import pickle

def clean_database():
    database_file = "face_database.pkl"
    
    if not os.path.exists(database_file):
        print("人脸库文件不存在，无需清理")
        return
    
    print("=" * 50)
    print("人脸库清理工具")
    print("=" * 50)
    
    # 读取旧数据库
    try:
        with open(database_file, 'rb') as f:
            old_faces = pickle.load(f)
    except Exception as e:
        print(f"读取数据库失败: {e}")
        return
    
    print(f"\n当前人脸库共 {len(old_faces)} 人:")
    for i, name in enumerate(old_faces.keys(), 1):
        print(f"  {i}. {name}")
    
    # 分离已注册的人和临时的人
    registered = {}
    temp = {}
    for name, embedding in old_faces.items():
        if name.startswith('Person_'):
            temp[name] = embedding
        else:
            registered[name] = embedding
    
    print(f"\n- 已注册的人: {len(registered)} 人")
    print(f"- 临时识别的人: {len(temp)} 人")
    
    if len(temp) == 0:
        print("\n没有需要清理的临时数据")
        return
    
    confirm = input("\n是否删除所有临时识别的人？(yes/no): ").strip().lower()
    
    if confirm == 'yes':
        # 备份旧文件
        backup_file = database_file + ".backup"
        import shutil
        shutil.copy2(database_file, backup_file)
        print(f"已备份到: {backup_file}")
        
        # 保存清理后的数据库
        try:
            with open(database_file, 'wb') as f:
                pickle.dump(registered, f)
            print(f"\n清理完成！现在人脸库共 {len(registered)} 人")
        except Exception as e:
            print(f"保存数据库失败: {e}")
    else:
        print("取消清理")

if __name__ == "__main__":
    clean_database()
