from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'/Users/appmacbook/Desktop/AIDM7340/yolov8n.pt')

# Define path to video file
source = r'/Users/appmacbook/Desktop/AIDM7340/sample/474288614-1-208.mp4'

# Run inference on the source
results = model(source, stream=True, show=True, save=True, save_conf=True, device='mps')  # generator of Results objects

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs