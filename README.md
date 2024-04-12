# AIDM7340

## This is the initial command and explanation of this Group Project

[Colab Running Command](https://colab.research.google.com/drive/1ZBsWqN9rYcsBEdkcdGXuaGF--IpOaEW6?usp=sharing)

The training of the model and the final presentation of the results are run in this colab document.

Below I will explain each piece of code in the colab documentation:

1. Copy everything from github to colab.
```python
!git clone https://github.com/LiXiaoxiaoSmile/AIDM7340.git
```

2. Runs all the required environments during model training.
```python
!pip install -r /content/AIDM7340/requirements.txt
```

3. Model training using already labelled datasets.
```python
!python /content/AIDM7340/train.py
```

4. Save the results of the trained optimal model.
```python
import os
import zipfile

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                                           os.path.join(folder_path, '..')))

folder_to_zip = '/content/runs'  # 要压缩的文件夹路径
zip_output = '/content/run.zip'  # 压缩文件输出路径
zip_folder(folder_to_zip, zip_output)
```

5. Applying models to video prediction.
```python
from ultralytics import YOLO
from PIL import Image
import torch

# Load a pretrained YOLOv8n model
model = YOLO(r'/content/AIDM7340/runs/detect/AIDM7340-GroupProject-2024-04-12-BestRun/weights/best.pt')

# Define path to video file
source = r'sample/474288614-1-208.mp4'

if torch.backends.mps.is_built():
    print("Apple Device GPU Detect, Using MPS for Speed Up.")
    device = 'mps'
elif torch.backends.cuda.is_built():
    print("CUDA Device Detect, Using CUDA for Speed Up.")
    device = 0
else:
    print("No Speed Up Device Detect, Using CPU. Caution: This may cause the Speed SlowDown.")
    device = 'cpu'


# Run inference on the source
results = model(source, stream=True, show=False, save=True, save_conf=True, device=device, imgsz=1280, retina_masks=True) #For Colab Use


for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

'''
# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f'results{i}.jpg')
'''
```