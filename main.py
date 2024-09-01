import cv2 
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


import cv2
import numpy as np
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,150)

classNames= []
classFile = '/content/labels.txt'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('n').split('n')

#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classLabels = []
file = 'labels.txt'
with open(file,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


img3 = cv2.imread('/Enter/the/path/of/your/image/')
plt.imshow(img3)

classIndex, confidence, bbox = net.detect(img3,confThreshold=0.6)
print(classIndex)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
# Convert bbox to a list so it can be iterated
for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img3,boxes,(255,0,0),2)
    cv2.putText(img3,classLabels[classInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
