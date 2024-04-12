from ultralytics import YOLO
from datetime import datetime
import torch

model = YOLO('yolov8n.pt')

local_train_config_filepath = r'./datasets/trainconfig-local.yaml'
colab_train_config_filepath = r'/content/AIDM7340/datasets/trainconfig-colab.yaml'

epoch = 100
imgsize = 1280

result_name = f'AIDM7340-GroupProject-{datetime.now()}'

if torch.backends.mps.is_built():
    print("Apple Device GPU Detect, Using MPS for Speed Up.")
    device = 'mps'
elif torch.backends.cuda.is_built():
    print("CUDA Device Detect, Using CUDA for Speed Up.")
    device = 0
else:
    print("No Speed Up Device Detect, Using CPU. Caution: This may cause the Speed SlowDown.")
    device = 'cpu'


try:
    results = model.train(data=local_train_config_filepath, epochs=epoch, imgsz=imgsize, device=device, name=result_name, workers=0)
except:
    print("Fail to load the local train config file, using Colab Version")
    results = model.train(data=colab_train_config_filepath, epochs=epoch, imgsz=imgsize, device=device, name=result_name, workers=0)
    
