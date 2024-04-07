from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')

results = model.train(data=r'/Users/appmacbook/Desktop/AIDM7340/datasets/trainconfig.yaml', epochs=100, imgsz=640, device='mps',
                      name='v8mtest', workers=0)