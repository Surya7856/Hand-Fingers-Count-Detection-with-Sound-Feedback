import time
import cv2
import torch
import pygame
import warnings
import os
cur=os.getcwd()
warnings.simplefilter("ignore")
model = torch.hub.load(os.path.join(cur,'yolov5'), 'custom', path=os.path.join(cur,'best(1).pt'), source='local')
cap = cv2.VideoCapture(0)
pygame.init()
pygame.mixer.init()
audio_file = ['Good Morning Student.mp3','How are you.mp3','I m Artificial Intel.mp3','I m Four.mp3','bye bye.mp3']
print(audio_file[0])
# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (500, 500))
    if not ret:
        print("Error: Failed to capture frame.")
        break
    results = model(frame)                  
    detections = results.pred[0]
    detected_labels = []

    for detection in detections:
        class_id = int(detection[-1])  # Get the class ID (the last value in detection tensor)
        confidence = detection[4].item()  # Confidence score of the detection
        label = model.names[class_id]  # Map class ID to class name (label)
        detected_labels.append((label, confidence))  # Add label and confidence to list
    for label, confidence in detected_labels:
        print(f"Class: {label}")
        pygame.mixer.music.load(audio_file[int(label)-1])
        pygame.mixer.music.play()
        time.sleep(3)
    # Annotate and display the frame with detected bounding boxes and class labels
    annotated_frame = results.render()[0]
    cv2.imshow('YOLOv5 Webcam', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
