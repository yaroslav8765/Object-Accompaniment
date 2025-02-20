
from ultralytics import YOLO, ASSETS
import numpy as np
import cv2


model = YOLO("yolo11n.pt", task="detection")


def detect_features(frame):
    result = model.track(frame, conf=0.3, iou=0.5)[0]
    for box in result.boxes.data: 
        x1, y1, x2, y2 = map(int, box[:4]) 
        conf = round(float(box[5]), 2)  
        cls = int(box[6]) 
        class_name = model.names[int(cls)]
        track_id = int(box[4])
        if conf >= 0.50:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
            cv2.rectangle(frame, (x1, y1), (x1+250, y1-30), (0, 0, 255), -1)
            cv2.putText(frame,f"Id{track_id} {class_name} {conf}",(x1,y1),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=1,thickness=1,color=(255,255,255))
    #if result.masks is not None:
    # for mask in result.masks:
    #    mask = mask.xy[0].astype(int) 
    #    mask_image = np.zeros_like(frame, dtype=np.uint8) 
    #    cv2.fillPoly(mask_image, [mask], (255, 255, 255)) 

    #    frame[mask_image[:, :, 0] > 0, 0] = 0 
    #    frame[mask_image[:, :, 1] > 0, 1] = 0  
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
    
    frame = detect_features(frame)

    cv2.imshow("Webcam!", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break  

cv2.destroyAllWindows()
stream.release() 


stream.release()
cv2.destroyAllWindows() #!