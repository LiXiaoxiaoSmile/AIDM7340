import os
import shutil
from sklearn.model_selection import train_test_split

def create_yolov9_dataset(src_img_dir, src_label_dir, dst_dir, test_size=0.2):
    # 创建目标目录结构
    os.makedirs(os.path.join(dst_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'labels', 'val'), exist_ok=True)

    # 获取所有图片文件名（不含扩展名）
    img_files = [f.split('.')[0] for f in os.listdir(src_img_dir) if f.endswith('.jpg')]

    # 划分训练集和验证集
    train_files, val_files = train_test_split(img_files, test_size=test_size, random_state=42)

    # 复制文件到目标目录
    for files, subset in zip([train_files, val_files], ['train', 'val']):
        for file in files:
            # 复制图片
            shutil.copy(os.path.join(src_img_dir, file + '.jpg'),
                        os.path.join(dst_dir, 'images', subset, file + '.jpg'))
            # 复制标签
            shutil.copy(os.path.join(src_label_dir, file + '.txt'),
                        os.path.join(dst_dir, 'labels', subset, file + '.txt'))

    print('Dataset created successfully.')

# 设置源目录和目标目录
src_img_dir = '/Users/muzian/Desktop/AIDM7340/raw_dataset/img'
src_label_dir = '/Users/muzian/Desktop/AIDM7340/raw_dataset/label'
dst_dir = '/Users/muzian/Desktop/AIDM7340/dataset'

# 创建数据集
create_yolov9_dataset(src_img_dir, src_label_dir, dst_dir)