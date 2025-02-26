
from ultralytics import YOLO, ASSETS
import numpy as np
import cv2

active_object_ID = 1
model = YOLO("yolo11n.pt")


def detect_objects(frame,active_object_ID):
    result = model.track(source = frame, conf=0.5, iou=0.5, show = False, persist=False)[0]
    num_objects = len(result.boxes.data)

    if(len(result)<=0):
        return frame
    else:
        for box in result.boxes.data: 
            if len(box) < 7:
                continue
            x1, y1, x2, y2 = map(int, box[:4]) 
            conf = round(float(box[5]), 2)  
            class_name = model.names[int(box[6])]
            track_id = int(box[4])
            draw_rectangle(frame, x1, y1, x2, y2, class_name, conf, track_id)
            cv2.putText(frame, f"Active object ID:{active_object_ID}", (frame.shape[0] - 150, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
            if active_object_ID == track_id:
                calculate_center_of_the_frame(frame.shape[0], frame.shape[1])
                calulate_center_of_the_object(x1, y1, x2, y2)
                cv2.line(frame, calculate_center_of_the_frame(frame.shape[1], frame.shape[0]), calulate_center_of_the_object(x1, y1, x2, y2), color=(0, 255, 0), thickness=2)
        
    return frame  

def calulate_center_of_the_object(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2

def calculate_center_of_the_frame(width, height):
    return width // 2, height // 2

def draw_rectangle(frame, x1, y1, x2, y2,class_name,conf,track_id):
    cv2.rectangle(frame, (x1, y1), (x2, y2),color= (255,0,0) if active_object_ID == track_id  else (0,0,255) ,thickness= 2)
    cv2.rectangle(frame, (x1, y1), (x1+250, y1-30), color= (255,0,0) if active_object_ID == track_id  else (0,0,255), thickness=-1)
    cv2.putText(frame,f"Id{track_id} {class_name} {conf}",(x1,y1),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=1,thickness=1,color=(255,255,255) )

def calculate_ditance_between_objects(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("No stream :(")
    exit()




while True:
    ret, frame = stream.read()
    if not ret:
        print("No more stream :(")
        break
    
    frame = detect_objects(frame, active_object_ID)
    #result = model.track(source = frame, persist = False)
    #anotate_frame = result[0].plot()
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
