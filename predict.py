from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'C:\Users\LQA\Desktop\class\Lxx\runs\detect\v8mtest3\weights\best.pt')

# Define path to video file
source = r'C:\Users\LQA\Desktop\class\Lxx\sample\474288614-1-16.mp4'

# Run inference on the source
results = model(source, stream=True, show=True, save=True, save_conf=True)  # generator of Results objects

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs