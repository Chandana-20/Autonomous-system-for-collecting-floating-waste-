import cv2
import matplotlib.pyplot as plt
import numpy as np
from gpiozero import LED
from time import sleep

in1 = LED(14)
in2 = LED(15)
in3 = LED(23)
in4 = LED(24)

config_file = r'/home/pi5/Downloads/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = r'/home/pi5/Downloads/frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classlabels = []
file_name = r'/home/pi5/Downloads/labels.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')
print(classlabels)

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

img = cv2.imread(r"/home/pi5/Downloads/WIN_20240127_14_56_20_Pro.jpg")
cv2.waitKey(100)

ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open the video')

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
zero_count = np.count_nonzero(ClassIndex != 0)

while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.65)

    if zero_count > 0:
        for Classind_array, conf, boxes in zip(ClassIndex, confidence, bbox):
            if (Classind_array == 44).any():
                cv2.rectangle(frame, boxes, (255, 255, 0), 2)
                cv2.putText(frame, classlabels[Classind_array - 1], (boxes[0] + 10, boxes[1] + 40),
                            font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
                cv2.putText(frame, str(conf), (boxes[0] + 10, boxes[1] + 200),
                            font, fontScale=font_scale, color=(0, 0, 255), thickness=3)
                print(Classind_array)
                x = boxes[0]
                y = boxes[1]
                w = boxes[2]
                h = boxes[3]
                x_center = x + w // 2
                y_center = y + h // 2
                print(x_center, y_center)
                if 250 < x_center < 400:
                    in1.on()
                    in2.off()
                    in3.off()
                    in4.on()
                elif x_center <= 250:
                    in1.off()
                    in2.off()
                    in3.off()
                    in4.on()
                elif x_center >= 400:
                    in1.on()
                    in2.off()
                    in3.off()
                    in4.off()
            else:
                in1.off()
                in2.off()
                in3.off()
                in4.off()

    cv2.imshow('Object Detection by Simple Learn', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
