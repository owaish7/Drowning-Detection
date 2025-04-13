# from ultralytics import YOLO
# import cv2

# # Load YOLO model
# model = YOLO("best_1.pt")  

# # Open webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

# DROWNING_CLASS_INDEX = 0  # Adjust based on your model's class index

# if not cap.isOpened():
#     print("Error: Webcam not detected or busy!")
# else:
#     print("Webcam detected successfully.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not capture frame!")
#             break

#         # Perform YOLO inference
#         results = model(frame)

#         drowning_detected = False

#         for r in results:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
#                 conf = box.conf[0].item()  # Confidence score
#                 cls = int(box.cls[0].item())  # Class index

#                 label = f"{model.names[cls]} {conf:.2f}"

#                 if cls == DROWNING_CLASS_INDEX:
#                     drowning_detected = True
#                     color = (0, 0, 255)  # Red for drowning alert
#                 else:
#                     color = (0, 255, 0)  # Green for other detections

#                 # Draw bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        
#         cv2.imshow("Drowning Detection", frame)

#         # Press 'q' to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()




# from ultralytics import YOLO
# import cv2
# import threading
# import pygame  # For playing sound

# # Initialize pygame mixer
# pygame.mixer.init()
# beep_sound_path = r"C:\yolo_test\267555__alienxxx__beep_sequence_02.wav"  # Update with your beep sound file path

# # Function to play the beep sound
# def play_beep():
#     pygame.mixer.music.load(beep_sound_path)
#     pygame.mixer.music.play()

# # Load YOLO model
# model = YOLO("best_1.pt")

# # Open webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# DROWNING_CLASS_INDEX = 0  # Adjust based on your model's class index

# if not cap.isOpened():
#     print("Error: Webcam not detected or busy!")
# else:
#     print("Webcam detected successfully.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not capture frame!")
#         break

#     # Perform YOLO inference
#     results = model(frame)

#     drowning_detected = False

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
#             conf = box.conf[0].item()  # Confidence score
#             cls = int(box.cls[0].item())  # Class index

#             label = f"{model.names[cls]} {conf:.2f}"

#             if cls == DROWNING_CLASS_INDEX:
#                 drowning_detected = True
#                 color = (0, 0, 255)  # Red for drowning alert
#             else:
#                 color = (0, 255, 0)  # Green for other detections

#             # Draw bounding box
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     # Play beep sound if drowning is detected
#     if drowning_detected:
#         threading.Thread(target=play_beep, daemon=True).start()  # Play sound in a separate thread

#     cv2.imshow("Drowning Detection", frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# pygame.mixer.quit()


# from ultralytics import YOLO
# import cv2
# import threading
# import time
# import pygame  # For playing sound
# import numpy as np
# import os

# # Check if files exist
# model_path = "best_1.pt"
# beep_sound_path = r"C:\yolo_test\267555__alienxxx__beep_sequence_02.wav"

# # Verify model file
# if not os.path.exists(model_path):
#     print(f"Error: Model file '{model_path}' not found!")
#     print(f"Current working directory: {os.getcwd()}")
#     print("Please place the model file in the correct location or update the path.")
#     exit()

# # Verify sound file
# if not os.path.exists(beep_sound_path):
#     print(f"Error: Sound file '{beep_sound_path}' not found!")
#     print("Using a dummy sound instead.")
#     # Create an alternative for the missing sound
#     beep_sound_path = None

# try:
#     # Initialize pygame mixer
#     pygame.mixer.init()
    
#     # Load sound if file exists
#     if beep_sound_path:
#         beep_sound = pygame.mixer.Sound(beep_sound_path)
#     else:
#         # Create a simple beep using pygame
#         pygame.mixer.Sound(pygame.sndarray.make_sound(np.sin(2*np.pi*np.arange(4000)/4000).astype(np.float32)))
        
#     # Class indices - make sure these match your model's class mapping
#     DROWNING_CLASS_INDEX = 0  # Class index for "drowning"
#     FLOATING_CLASS_INDEX = 1  # Class index for "floating"
    
#     # Load YOLO model with try-except to catch errors
#     print("Loading YOLO model...")
#     try:
#         model = YOLO(model_path)
#         print("Model loaded successfully")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         exit()
    
#     # Set confidence thresholds
#     DROWNING_CONF_THRESHOLD = 0.40  # Higher threshold for drowning class
#     FLOATING_CONF_THRESHOLD = 0.30  # Lower threshold for floating class
    
#     # Setup for alert cooldown
#     last_alert_time = 0
#     ALERT_COOLDOWN = 2  # seconds between alerts
    
#     # Try multiple camera indices if the first one fails
#     cap = None
#     for cam_id in [0, 1, 2]:
#         try:
#             cap = cv2.VideoCapture(cam_id)
#             if cap.isOpened():
#                 print(f"Webcam detected successfully on index {cam_id}.")
#                 break
#         except:
#             continue
    
#     if cap is None or not cap.isOpened():
#         print("Error: Could not open any webcam!")
#         exit()
        
#     # Set camera properties
#     try:
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         cap.set(cv2.CAP_PROP_FPS, 30)
#     except Exception as e:
#         print(f"Warning: Could not set camera properties: {e}")
    
#     # Frame skipping for better performance
#     process_every_n_frames = 2
#     frame_count = 0
    
#     # FPS calculation
#     fps_start_time = time.time()
#     fps_frame_count = 0
#     fps = 0
    
#     print("Starting detection...")
#     while True:
#         try:
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 print("Error: Could not capture frame!")
#                 # Try to reinitialize camera
#                 cap.release()
#                 cap = cv2.VideoCapture(0)
#                 if not cap.isOpened():
#                     break
#                 continue
            
#             # Skip frames for better performance
#             frame_count += 1
#             if frame_count % process_every_n_frames != 0:
#                 # Still display the frame but don't run detection
#                 cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 cv2.imshow("Drowning Detection", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#                 continue
            
#             # Perform YOLO inference with error handling
#             try:
#                 results = model(frame)
#             except Exception as e:
#                 print(f"Error during inference: {e}")
#                 cv2.putText(frame, "Inference Error", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 cv2.imshow("Drowning Detection", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#                 continue
            
#             # Track if drowning was detected in this frame
#             drowning_detected = False
            
#             # Process results with error handling
#             try:
#                 for r in results:
#                     for box in r.boxes:
#                         # Get box details
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
#                         w, h = x2 - x1, y2 - y1  # Calculate width and height
#                         conf = float(box.conf[0])  # Confidence score - using float instead of .item()
#                         cls = int(box.cls[0])  # Class index - using int instead of .item()
                        
#                         # Apply size filtering - ignore very small or very large detections
#                         min_size = 20  # Minimum width/height in pixels
#                         max_size = 400  # Maximum width/height in pixels
#                         if w < min_size or h < min_size or w > max_size or h > max_size:
#                             continue
                            
#                         # Apply class-specific confidence thresholds
#                         if cls == DROWNING_CLASS_INDEX and conf < DROWNING_CONF_THRESHOLD:
#                             continue
#                         elif cls == FLOATING_CLASS_INDEX and conf < FLOATING_CONF_THRESHOLD:
#                             continue
                        
#                         # Get class name and prepare label
#                         class_name = model.names[cls] if cls in model.names else f"Class {cls}"
#                         label = f"{class_name} {conf:.2f}"
                        
#                         # Set color based on class (drowning = red, floating = green)
#                         if cls == DROWNING_CLASS_INDEX:
#                             drowning_detected = True
#                             color = (0, 0, 255)  # Red for drowning
#                         else:
#                             color = (0, 255, 0)  # Green for floating
                        
#                         # Draw bounding box and label
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
#                         # Draw label with background
#                         (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#                         cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
#                         cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             except Exception as e:
#                 print(f"Error processing detections: {e}")
#                 cv2.putText(frame, "Detection Error", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
#             # Calculate and display FPS
#             fps_frame_count += 1
#             if fps_frame_count >= 10:
#                 end_time = time.time()
#                 fps = fps_frame_count / (end_time - fps_start_time)
#                 fps_start_time = end_time
#                 fps_frame_count = 0
            
#             # Display FPS on frame
#             cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
#             # Play alert sound with cooldown if drowning is detected
#             current_time = time.time()
#             if drowning_detected and (current_time - last_alert_time > ALERT_COOLDOWN):
#                 if beep_sound_path:
#                     beep_sound.play()
#                 last_alert_time = current_time
                
#                 # Display visual alert
#                 cv2.putText(frame, "DROWNING ALERT!", (frame.shape[1]//2 - 150, 50), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
#             # Show the frame
#             cv2.imshow("Drowning Detection", frame)
            
#             # Press 'q' to exit
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#         except Exception as e:
#             print(f"Unexpected error: {e}")
#             continue

# except Exception as e:
#     print(f"Fatal error: {e}")

# finally:
#     # Clean up
#     print("Shutting down...")
#     if 'cap' in locals() and cap is not None:
#         cap.release()
#     cv2.destroyAllWindows()
#     pygame.quit()
#     print("Program terminated.")



from ultralytics import YOLO
import cv2
import threading
import pygame  # For playing sound

# Initialize pygame mixer
pygame.mixer.init()
beep_sound_path = r"C:\yolo_test\267555__alienxxx__beep_sequence_02.wav"  # Update with your beep sound file path

# Function to play the beep sound
def play_beep():
    pygame.mixer.music.load(beep_sound_path)
    pygame.mixer.music.play()

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

    # Play beep sound if drowning is detected
    if drowning_detected:
        threading.Thread(target=play_beep, daemon=True).start()  # Play sound in a separate thread

    cv2.imshow("Drowning Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
