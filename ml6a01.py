import time
import cv2
from Lora import lora
from box_count import box_count

cfg = "yolov4.cfg"
weights = "yolov4.weights"
net = cv2.dnn_DetectionModel(cfg,weights)
net.setInputSize(512,512)
net.setInputScale(1.0/255)
net.setInputSwapRB(True)

"""
with open("obj.names","r") as f: names = f.read().rstrip('\n').split('\n')
print(names)
"""

def image_detection():
    
    width = 1920#darknet.network_width(network)
    height = 1080#darknet.network_height(network)    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    for _ in range(5):cap.read()
    image = cap.read()[1]
    
    # Uncomment next line for local image detection
    #image = cv2.imread("test.jpg")
    cls,conf,boxes = net.detect(image,confThreshold=0.1,nmsThreshold=0.4)
    box_num = box_count(cls,conf,boxes) if len(cls)!=0 else [0,0]
    cap.release()
    return box_num

def LORA():
    _lora = lora()

    while True:
        # Do something before next detection
        box1,box2 = image_detection()
        _lora.send(f"Area1: {box1}, Area2: {box2}")
        print(box1,box2)
        
        # Enter to continue
        #print("Press 'Enter' to continue")
        #a= input()
        # Sleep 1 hour
        print("Sleep 1 hour......")
        time.sleep(3600)
def draw_boxes(image,cls,color,boxes):
    x,y = boxes[0],boxes[1]
    w,h = boxes[2],boxes[3]
    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        
def SHOW():
    #cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    #while True:
    if True:
        image = cv2.imread("test.jpg")
        width = 1920
        height = 1080
        cls,conf,boxes = net.detect(image,confThreshold=0.1,nmsThreshold=0.4)
        for i in range(len(conf)):
            if conf[i][0]>0.8: # If confidence greater than 80%, draw its box.
                # Box top
                if cls[i][0] == 0:
                    draw_boxes(image,"BOX TOP",(255,0,0),boxes[i])
                else:
                    draw_boxes(image,"BOX",(0,255,0),boxes[i])
        cv2.imshow("img",image)
        cv2.waitKey(0)
    #cap.release()
if __name__ == "__main__":
    #LORA()
    SHOW()