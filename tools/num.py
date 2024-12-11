import os


def count_categories(image_folder, label_folder):
    # 初始化计数器
    category_counts = {'norm': 0, 0: 0, 1: 0, 2: 0, 3: 0}
    img_files = set()
    label_files = set()

    # 获取图像文件和标签文件列表
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.png'):  # 假设图片文件是png格式
                img_files.add(file)

    for root, dirs, files in os.walk(label_folder):
        for file in files:
            if file.endswith('.txt'):  # 假设标签文件是txt格式
                label_files.add(file)

    # 遍历所有图片，统计类别
    for img_file in img_files:
        # 检查对应标签文件是否存在
        label_file = os.path.splitext(img_file)[0] + '.txt'

        if label_file not in label_files:
            # 如果没有标签文件，则标记为 'norm'
            category_counts['norm'] += 1
        else:
            # 如果标签文件存在，读取标签文件，检查类别
            with open(os.path.join(label_folder, label_file), 'r') as f:
                labels = f.readlines()
                categories_in_file = set()
                for label in labels:
                    category = int(label.strip().split()[0])  # 假设类别是标签文件的第一个数字
                    if category in category_counts:
                        categories_in_file.add(category)
                for category in categories_in_file:
                    category_counts[category] += 1

    # 输出统计结果
    print(f"norm的数量: {category_counts['norm']}")
    for category in range(4):
        print(f"类别 {category} 的数量: {category_counts[category]}")
    print(f"图像文件的数量: {len(img_files)}")


# 设置图片和标签文件夹路径
image_folder = r'E:\yolo\data_best\images\test'  # 替换为图像文件夹路径
label_folder = r'E:\yolo\data_best\labels\test'  # 替换为标签文件夹路径

# 运行统计
count_categories(image_folder, label_folder)
