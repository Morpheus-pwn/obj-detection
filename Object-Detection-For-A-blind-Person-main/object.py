import cv2 as cv
import pyttsx3 

text= pyttsx3.init()

cap=cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames= []
classFile = "coco.names"
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

weightsPath = 'frozen_inference_graph.pb'

net = cv.dnn_DetectionModel(weightsPath,configPath)

net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5) 
net. setInputMean ((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    ret,img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv.rectangle(img,box,color=(0,255,0),thickness=2)
            cv.putText(img, classNames[classId-1].upper(), (box [0]+10,box[1]+30),
            cv.FONT_HERSHEY_COMPLEX,1,(0,255.0),2) 
    cv.imshow("output",img)
    cv.waitKey(1)
    ans= classNames[classId-1]
    text.setProperty('rate', 125)
    text.say(ans)
    text.runAndWait()
