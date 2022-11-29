import torch
import cv2

model_name="Trained_model/yolov5/weights/best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_name, force_reload=True)
frame = cv2.imread("test/images/1.jpg")

y_shape, x_shape = frame.shape[0], frame.shape[1]

result = model(frame)
#print(result.xyxyn[0])

labels, cord = result.xyxyn[0][:, -1], result.xyxyn[0][:, :-1]
for i in range(len(cord)):
    row = cord[i]
    x1, y1, x2, y2 = row[0]*x_shape, row[1]*y_shape, row[2]*x_shape, row[3]*y_shape
    print(x1,y1,x2,y2)
#print(labels, cord)


label, cor = result.xywhn[0][:, -1], result.xywhn[0][:, :-1]
for i in range(len(label)):
    row = cor[i]
    x, y, w, h = row[0]*x_shape, row[1]*y_shape, row[2]*x_shape, row[3]*y_shape
    print(x,y,w,h)