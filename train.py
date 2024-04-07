from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')

results = model.train(data=r'C:\Users\LQA\Desktop\class\Lxx\dataset\dataset.yaml', epochs=100, imgsz=640, device=0,
                      name='v8mtest', workers=0)