
from ultralytics import YOLO, ASSETS
import numpy as np
import cv2
import serial 

active_object_ID = 1
model = YOLO("yolo11n.pt")
ser = serial.Serial(port = 'COM7',baudrate= 9600)
click_position = None
last_object_position = None
last_object_class = None

def detect_objects(frame):
    global last_object_position  
    global last_object_class
    global click_position
    result = model.track(source=frame, conf=0.3, iou=0.5, show=False, persist=False)[0]

    if not hasattr(result.boxes, "data") or result.boxes.data is None or len(result.boxes.data) == 0:
        last_object_position = None 
        return frame

    detected_objects = []
    
    for box in result.boxes.data :
        if len(box) < 7:
            continue

        x1, y1, x2, y2 = map(int, box[:4])
        track_id = int(box[4])
        class_id = int(box[6])
        
        if class_id not in model.names:
            continue

        class_name = model.names[class_id]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        detected_objects.append((track_id, class_name, (center_x, center_y), (x1, y1, x2, y2)))

    if last_object_position is not None and click_position is None:
        min_distance = float("inf")
        best_match = None
        for obj in detected_objects:
            obj_center = obj[2]
            distance = calculate_ditance_between_objects(last_object_position[0], last_object_position[1], obj_center[0], obj_center[1])
            if distance < min_distance and obj[1] == last_object_class:
                min_distance = distance
                best_match = obj
    
        if best_match is not None:
            object_id, class_name, center, bbox = best_match
            last_object_position = center
            last_object_class = class_name
            draw_rectangle(frame, *bbox, class_name, 1.0, object_id)
            cv2.line(frame, last_object_position, calculate_center_of_the_frame(frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
                
    else:
        if(click_position is not None):
            min_distance = float("inf")
            best_match = None
            last_object_position = None
            last_object_class = None
            for obj in detected_objects:
                obj_center = obj[2]
                distance = calculate_ditance_between_objects(click_position[0], click_position[1], obj_center[0], obj_center[1])
                if distance < min_distance:
                    min_distance = distance
                    best_match = obj

            if best_match is not None:
                object_id, class_name, center, bbox = best_match
                last_object_position = center
                last_object_class = class_name
                draw_rectangle(frame, *bbox, class_name, 1.0, object_id)
                cv2.line(frame, last_object_position, calculate_center_of_the_frame(frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
            click_position = None
    
    cv2.putText(frame, f"Click {click_position}", (frame.shape[0] - 150, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

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

def mouse_callback(event, x, y, flags, param):
    global click_position
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)


stream = cv2.VideoCapture(0)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)
if not stream.isOpened():
    print("No stream :(")
    exit()




while True:

    ser.write(b"Weee\n")
    value = ser.readline()
    read_str = str(value, 'UTF-8')
    print(read_str)
    ret, frame = stream.read()
    if not ret:
        print("No more stream :(")
        break
    
    frame = detect_objects(frame)
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) 

    if key == ord('w'):
        active_object_ID += 1
    elif key == ord('s'):
        active_object_ID -= 1
        active_object_ID = 1 if active_object_ID <= 0 else active_object_ID
    elif key == ord('q'):
        break  

cv2.destroyAllWindows()
stream.release() 
