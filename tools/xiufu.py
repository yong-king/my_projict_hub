import json


def fix_empty_bbox(json_file, output_json):
    # 加载 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 修复空 bbox 的 annotation
    annotations = data.get('annotations', [])
    fixed_annotations = []
    for ann in annotations:
        if 'bbox' in ann and (not ann['bbox'] or len(ann['bbox']) != 4):
            # 修复为空或不完整的 bbox
            print(f"修复 annotation id={ann['id']} 中的空 bbox")
            ann['bbox'] = [0, 0, 1, 1]  # 使用默认值修复（可以根据实际需求调整）
        fixed_annotations.append(ann)

    data['annotations'] = fixed_annotations

    # 保存修复后的 JSON 文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"修复完成！修复后的文件保存为 {output_json}")


# 使用示例
json_file = r'E:\YSH\mmdetection-main\mmdetection-main\data\coco\annotations\instances_val2017.json'  # 输入的 JSON 文件路径
output_json = r'E:\YSH\mmdetection-main\mmdetection-main\data\coco\annotations\instances_val2017.json'  # 修复后的 JSON 文件路径

fix_empty_bbox(json_file, output_json)
