from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("best.pt")  

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

DROWNING_CLASS_INDEX = 0  # Adjust based on your model's class index

if not cap.isOpened():
    print("Error: Webcam not detected or busy!")
else:
    print("Webcam detected successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame!")
            break

        # Perform YOLO inference
        results = model(frame)

        drowning_detected = False

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index

                label = f"{model.names[cls]} {conf:.2f}"

                if cls == DROWNING_CLASS_INDEX:
                    drowning_detected = True
                    color = (0, 0, 255)  # Red for drowning alert
                else:
                    color = (0, 255, 0)  # Green for other detections

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        
        cv2.imshow("Drowning Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
