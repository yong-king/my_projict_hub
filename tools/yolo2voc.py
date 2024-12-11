import os
import xml.etree.ElementTree as ET
from PIL import Image
import shutil
from tqdm import tqdm
import argparse


def create_voc_xml(image_path, boxes, labels, out_path):
    img = Image.open(image_path)
    width, height = img.size

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = os.path.basename(os.path.dirname(os.path.dirname(out_path)))
    ET.SubElement(root, "filename").text = os.path.basename(image_path)

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(3)  # Assuming RGB images

    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(box[0]))
        ET.SubElement(bbox, "ymin").text = str(int(box[1]))
        ET.SubElement(bbox, "xmax").text = str(int(box[2]))
        ET.SubElement(bbox, "ymax").text = str(int(box[3]))

    tree = ET.ElementTree(root)
    tree.write(out_path)


def yolo_to_voc_format(yolo_box, img_width, img_height):
    x_center, y_center, w, h = yolo_box
    w *= img_width
    h *= img_height
    x_center *= img_width
    y_center *= img_height

    xmin = int(x_center - w / 2)
    ymin = int(y_center - h / 2)
    xmax = int(x_center + w / 2)
    ymax = int(y_center + h / 2)

    return [xmin, ymin, xmax, ymax]


def convert_dataset(img_path, label_path, output_path, image_set):
    os.makedirs(os.path.join(output_path, "Annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "ImageSets", "Main"), exist_ok=True)

    image_list = []

    for img_file in tqdm(os.listdir(img_path), desc=f"Processing {image_set}"):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            img_filename = os.path.splitext(img_file)[0]
            image_list.append(img_filename)

            # Copy image
            shutil.copy(
                os.path.join(img_path, img_file),
                os.path.join(output_path, "JPEGImages", img_file)
            )

            # Process label
            label_file = os.path.join(label_path, f"{img_filename}.txt")
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                img = Image.open(os.path.join(img_path, img_file))
                img_width, img_height = img.size

                boxes = []
                labels = []
                for line in lines:
                    data = line.strip().split()
                    label = data[0]
                    box = list(map(float, data[1:]))
                    voc_box = yolo_to_voc_format(box, img_width, img_height)
                    boxes.append(voc_box)
                    labels.append(label)

                # Create XML annotation
                xml_path = os.path.join(output_path, "Annotations", f"{img_filename}.xml")
                create_voc_xml(os.path.join(img_path, img_file), boxes, labels, xml_path)

    # Create ImageSets file
    with open(os.path.join(output_path, "ImageSets", "Main", f"{image_set}.txt"), 'w') as f:
        for item in image_list:
            f.write(f"{item}\n")


def main(train_img_path, train_label_path, val_img_path, val_label_path, output_path):
    # Convert training set
    convert_dataset(train_img_path, train_label_path, output_path, "train")

    # Convert validation set
    convert_dataset(val_img_path, val_label_path, output_path, "val")

    print("转换完成！VOC格式数据集已保存到", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO format dataset to VOC format")
    parser.add_argument("--train_img", default="E:/Datasets/UCAS-AOD-yolo/train/images", help="Path to training images")
    parser.add_argument("--train_label", default="E:/Datasets/UCAS-AOD-yolo/train/labels",
                        help="Path to training labels")
    parser.add_argument("--val_img", default="E:/Datasets/UCAS-AOD-yolo/val/images", help="Path to validation images")
    parser.add_argument("--val_label", default="E:/Datasets/UCAS-AOD-yolo/val/labels", help="Path to validation labels")
    parser.add_argument("--output", default="E:/Datasets/VOCdata/VOC-USCA-AOD", help="Path to output VOC dataset")

    args = parser.parse_args()

    main(args.train_img, args.train_label, args.val_img, args.val_label, args.output)