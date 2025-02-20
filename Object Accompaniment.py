
from ultralytics import YOLO, ASSETS
import numpy as np
import cv2

active_object_ID = 1
model = YOLO("yolo11n.pt", task="detection")


def detect_features(frame,active_object_ID):
    result = model.track(frame, conf=0.5, iou=0.5)[0]
    num_objects = len(result.boxes.data)
    if active_object_ID > num_objects:
        active_object_ID = num_objects
    #test comment##%$#%#%#%@@%%@%537457git 
    
    for box in result.boxes.data: 
        x1, y1, x2, y2 = map(int, box[:4]) 
        conf = round(float(box[5]), 2)  
        cls = int(box[6]) 
        class_name = model.names[int(cls)]
        track_id = int(box[4])
        if conf >= 0.50:
            cv2.rectangle(frame, (x1, y1), (x2, y2),color= (255,0,0) if active_object_ID == track_id  else (0,0,255) ,thickness= 2)
            cv2.rectangle(frame, (x1, y1), (x1+250, y1-30), color= (255,0,0) if active_object_ID == track_id  else (0,0,255), thickness=-1)
            cv2.putText(frame,f"Id{track_id} {class_name} {conf}",(x1,y1),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=1,thickness=1,color=(255,255,255) )
    return frame  





stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("No stream :(")
    exit()

fps = stream.get(cv2.CAP_PROP_FPS)
width = int(stream.get(3))
height = int(stream.get(4))


while True:
    ret, frame = stream.read()
    if not ret:
        print("No more stream :(")
        break
    
    frame = detect_features(frame, active_object_ID)

    cv2.imshow("Webcam!", frame)
    
    key = cv2.waitKey(1) 

    if key == ord('w'):
        active_object_ID += 1
    elif key == ord('s'):
        active_object_ID -= 1
        active_object_ID = 1 if active_object_ID <= 0 else active_object_ID
    elif cv2.waitKey(1) == ord('q'):
        break  

cv2.destroyAllWindows()
stream.release() 


stream.release()
cv2.destroyAllWindows() #!