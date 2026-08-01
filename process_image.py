# import necessary libraries
import cv2
from ultralytics import YOLO

# initialize the detector with weights
detector = YOLO("./model/traffic_sign_detector.pt", task="detect")

# path to the image
img_path = "./data/input/stop_sign.jpg"

# to visualize detections

# reading the image
image = cv2.imread(img_path)

# detector.predict returns a list of detection objects
detections = detector.predict(image)

for detection in detections:
    # Get class indices and class names
    class_ids = detection.boxes.cls  # cls stores the class IDs
    
    # Iterate over bounding boxes
    for i, bbox in enumerate(detection.boxes):
        # Get the bounding box coordinates
        x1, y1, x2, y2 = bbox.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Get class name using the class ID
        class_id = int(class_ids[i])
        class_name = detection.names[class_id]  # detection.names holds the class names
        
        # Display the class name on the image
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Show the image with detection
        cv2.imshow("Detected", image)
        
        # Break on 'q' key
        if cv2.waitKey(10000) & 0xFF == ord("q"):
            break

# uncomment the following line to detect and save the annotated image without visualizing it
# results = detector(img_path, save=True)