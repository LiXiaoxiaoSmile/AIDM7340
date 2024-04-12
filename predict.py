from ultralytics import YOLO
from PIL import Image
import torch

# Load a pretrained YOLOv8n model
model = YOLO(r'runs/detect/AIDM7340-GroupProject-2024-04-12-BestRun/weights/best.pt')

# Define path to video file
source = r'sample/328907506-1-16.mp4'

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
results = model(source, stream=True, show=True, save=True, save_conf=True, device=device, imgsz=1280, visualize=False, retina_masks=True)  # generator of Results objects
# results = model(source, stream=True, show=False, save=True, save_conf=True, device=device, imgsz=1280, visualize=False, retina_masks=True) # Colab cannot show the demo


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