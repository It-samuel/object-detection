import cv2
import imutils
#load dnn
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.cfg", "dnn_model/yolov4-tiny.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(500, 500), scale=1/255)

#loadng class objects
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

cam = cv2.VideoCapture(0)


while True:
    _, frame = cam.read()
    frame= imutils.resize(frame,height=450, width=900)
    #object detection
    (class_id, scores, bboxes) = model.detect(frame)
    for class_id, scores, bboxes in zip (class_id, scores, bboxes):
        (x, y, w, h) = bboxes
        class_name = classes[class_id]

        cv2.putText(frame, class_name, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (200, 0, 50), 4)

    cv2.imshow('object detection', frame)
    cv2.waitKey(1)