from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\yolo\v8\ultralytics-8.2.87\yolo\pretrain98\weights\best.pt') # 自己训练结束后的模型权重
    model.val(data=r'E:\yolo\v8\ultralytics-8.2.87\data.yaml',
              split='test',
              imgsz=320,
              batch=64,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
