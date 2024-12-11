import os
import json
from PIL import Image

# 文件夹路径
image_folder = r"E:\YSH\mmdetection-main\mmdetection-main\data\coco\test2017 - 副本\images"  # 存放图片的文件夹路径
label_folder = r"E:\YSH\mmdetection-main\mmdetection-main\data\coco\test2017 - 副本\labels"  # 存放YOLO标签的文件夹路径
output_file = r"E:\YSH\mmdetection-main\mmdetection-main\data\coco\annotations\instances_test2017.json"  # 输出文件路径

# 初始化 COCO 格式的字典
coco_format = {
    "images": [],
    "annotations": [],
    "categories": []
}

# 定义类别
categories = [
    {"id": 1, "name": "category1"},
    {"id": 2, "name": "category2"},
    {"id": 3, "name": "category3"},
    {"id": 4, "name": "category4"}
]
coco_format["categories"].extend(categories)

# 获取所有图片和标签文件
image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))}
label_files = {os.path.splitext(f)[0]: f for f in os.listdir(label_folder) if f.endswith('.txt')}

# 全局 ID 累计
image_id = 0
annotation_id = 0

# 记录未匹配情况
images_without_labels = []  # 没有标签的图片
labels_without_images = []  # 没有对应图片的标签
images_with_empty_labels = []  # 有标签但标签文件为空的图片

for image_name, image_file in image_files.items():
    image_id += 1
    image_path = os.path.join(image_folder, image_file)

    # 获取图片的宽度和高度
    with Image.open(image_path) as img:
        width, height = img.size

    # 添加到 images
    coco_format["images"].append({
        "file_name": image_file,
        "width": width,
        "height": height
    })

    # 如果有对应的标签文件，则处理标签
    if image_name in label_files:
        label_path = os.path.join(label_folder, label_files[image_name])
        with open(label_path, "r") as f:
            lines = f.readlines()
            if not lines:
                images_with_empty_labels.append(image_file)
            for line in lines:
                try:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

                    # 转换 YOLO 格式到 COCO 格式
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    bbox_width = bbox_width * width
                    bbox_height = bbox_height * height

                    # 计算面积
                    area = bbox_width * bbox_height

                    # 添加到 annotations
                    annotation_id += 1
                    coco_format["annotations"].append({
                        "image_id": image_id,
                        "category_id": int(class_id) + 1,  # 假设类别从1开始
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": area,
                        "iscrowd": 0
                    })
                except ValueError:
                    print(f"标签文件格式错误: {label_path}")
    else:
        images_without_labels.append(image_file)

# 找出没有对应图片的标签
unused_labels = set(label_files.keys()) - set(image_files.keys())
for label_name in unused_labels:
    labels_without_images.append(label_files[label_name])

# 打印未匹配的信息
if images_without_labels:
    print("没有标签的图片如下：")
    print("\n".join(images_without_labels))

if labels_without_images:
    print("没有对应图片的标签如下：")
    print("\n".join(labels_without_images))

if images_with_empty_labels:
    print("标签文件为空的图片如下：")
    print("\n".join(images_with_empty_labels))

# 保存为 JSON 文件
with open(output_file, "w") as json_file:
    json.dump(coco_format, json_file, indent=4)

print(f"转换完成！COCO格式文件保存在：{output_file}")
