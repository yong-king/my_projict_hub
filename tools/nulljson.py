import os
import json


def check_images_and_labels(image_folder, json_file, output_json):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取 JSON 中已有的图片文件名
    existing_images = {image['file_name'] for image in data.get('images', [])}

    # 获取文件夹中的所有图片文件名
    all_images = {img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))}

    # 找出图片中没有在 JSON 文件中的
    missing_images_in_json = all_images - existing_images

    # 找出 JSON 文件中没有在图片文件夹中的
    missing_images_in_folder = existing_images - all_images

    # 打印缺失的图片
    if missing_images_in_json:
        print("以下图片不在 JSON 文件中，需要添加空标签：")
        for img in missing_images_in_json:
            print(img)
    else:
        print("所有图片都已经在 JSON 文件中，无需添加空标签。")

    # 打印没有对应图片的标签
    if missing_images_in_folder:
        print("\n以下标签在 JSON 文件中，但对应的图片文件不存在：")
        for img in missing_images_in_folder:
            print(img)
    else:
        print("\nJSON 文件中的所有标签都有对应的图片文件。")

    # 如果有缺失的图片，添加空标签
    for missing_image in missing_images_in_json:
        # 获取新的 image_id
        new_image_id = len(data['images']) + 1

        # 添加图片信息
        data['images'].append({
            "id": new_image_id,
            "file_name": missing_image,
            "width": 0,  # 如果有宽度、高度信息，可以填写实际值
            "height": 0
        })

        # 添加空的注释（标签）
        data['annotations'].append({
            "id": len(data['annotations']) + 1,
            "image_id": new_image_id,
            "category_id": 0,  # 假设 0 代表空标签类别
            "bbox": [],  # 空边界框
            "segmentation": [],  # 空分割
            "area": 0
        })

    # 保存新的 JSON 文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n处理完成！缺失的图片已添加空标签，结果保存到 {output_json}")


# 使用示例
image_folder = r'E:\YSH\mmdetection-main\mmdetection-main\data\coco\val2017\images'  # 图片文件夹路径
json_file = r'E:\YSH\mmdetection-main\mmdetection-main\data\coco\val2017\annotations\val.json'  # 输入的 JSON 文件路径
output_json = r'E:\YSH\mmdetection-main\mmdetection-main\data\coco\val2017\annotations\val.json'  # 输出的 JSON 文件路径

check_images_and_labels(image_folder, json_file, output_json)
