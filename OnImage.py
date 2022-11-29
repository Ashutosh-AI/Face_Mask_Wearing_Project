import json
import torch
import numpy as np
import cv2
from time import time
import os
import glob

class MaskDetection:
    """
    Class implements Yolo model to make inferences on a Youtube video using Opencv2.
    """

    def __init__(self, capture_index, model_name):
        """
        :param capture_index: index value 0 for WebCam, or pass a video file
        :param model_name: Trained model
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)


    def get_image_capture(self):
        """
        Creates a new video streaming boject to extract video frame by frame to make prediction on.
        :return: Opencv2 video capure object
        """
        images = glob.glob(self.capture_index)
        return images



    def load_model(self, model_name):
        """
        Load Yolov5 model from pytorch hub.
        :return:  Trained Pytorch model
        """
        if model_name:
            model = torch.hub.load("ultralytics/yolov5", "custom", path=model_name, force_reload=True)
        else:
            model = torch.hub.load("ultralytics/yolov5","yolov5s", pretrained=True)
        return model


    def score_frame(self, frame):
        """
        Takes a single frame as input, and score the frame using yolov5 model
        :param frame: input frame numpy/list/touple format.
        :return: Label and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def get_xyxy2xywh(self, x1, y1, x2, y2):
        """
        :param x1: predicted xmin
        :param y1: predicted ymin
        :param x2: predicted xmax
        :param y2: predicted ymax
        :return: center coordinate (x,y) and width, height
        """

        x_center = ((x1 + x2) / 2)
        y_center = ((y1 + y2) / 2)
        width = (x2 - x1)
        height = (y2 - y1)
        bbox = x_center, y_center, width, height
        return(bbox)

    def plot_boxes(self, results, frame, imageId, result):
        """
        Takes a frame and its result as input, and plot the bounding boxes and labels on the frame.
        :param results: contains label and coordinates predicted by model on the given frame
        :param frame:frame which has been scored.
        :return: frame with bounding boxes and labels ploted on it
        """

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.4:   # Threshold
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0,255,0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr)
                cv2.putText(frame, str(np.round(float(row[4]), 2)), (x1+80,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr)


                #Save result in coco format for evaluate mAP
                result.append({
                    "image_id": imageId,
                    "category_id": int(labels[i]),  # Tensor to integer
                    "bbox": self.get_xyxy2xywh(x1, y1, x2, y2),
                    "score": float(np.round(float(row[4]), 2))  # Tensor to float
                })
                print()
                with open("Pred_result", "w") as f:
                    json.dump(result, f, indent=4)

        return frame


    def __call__(self):

        images = self.get_image_capture()
        result = []

        #assert cap.isOpened()
        for frame in images:
            imageId = int(os.path.splitext(os.path.basename(frame))[0])
            print(imageId)
            frame = cv2.imread(frame)

            #print(frame)
            frame = cv2.resize(frame, (416, 416))

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame, imageId, result)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            # print(f"Frames Per Second : {fps}")

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow("YOLOv5 Detection", frame)
            cv2.waitKey(0)
        images.release()


# Create a new object and execute
detector = MaskDetection(capture_index="test/images/*.jpg", model_name="Trained_model/yolov5/weights/best.pt")
detector()
