#Import the neccesary libraries
import numpy as np
import argparse
import cv2 
import time
import sys

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='ISIDS 2')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD v2 on Coco for Tensor model or '
                                       )
parser.add_argument('-O', '--nonmax', required=False, action='store_true', default=False,
                help = 'flag to enable non-max suppression (disabled by default)')
parser.add_argument('-S', '--skip', required=False, action='store_true', default=False,
                help = 'flag to disable detection (for checkin real fps)')
parser.add_argument("--weights", default="models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
                                 help='Path to weights: '
                                      'MobileNetSSD v2 on Coco for Tensor model or '
                                      )
parser.add_argument("--thr", default=0.5, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# function fro drawing boxes around detencted objects
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classNames[class_id]) + ": %.2f"%confidence
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Labels of Network.
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

# Open video file or capture device. 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)


# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))

#Load the Caffe model 
net = cv2.dnn.readNetFromTensorflow(args.weights,args.prototxt)

while True:
    start = time.time() 
    # Capture frame-by-frame
    ret, frame = cap.read()
    if (not args.skip):
      resized = cv2.resize(frame, (300, 300))
      rows, cols, channels = frame.shape

      # MobileNet requires fixed dimensions for input image(s)
      # so we have to ensure that it is resized to 300x300 pixels.
      # set a scale factor to image because network the objects has differents size. 
      # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
      # after executing this command our "blob" now has the shape:
      # (1, 3, 300, 300)
      blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
      #Set to network the input blob 
      net.setInput(blob)
      #Prediction of network
      outs = net.forward()

      class_ids = []
      confidences = []
      boxes = []
      # loop over all detected objects
      for detection in outs[0, 0, :, :]:        
          # get the score
          score = float(detection[2])
          # draw the bounding box
          if score > args.thr: 
              # get the class index
              class_id=int(detection[1])
              class_ids.append(class_id)
              confidences.append(score)
              left   = int(detection[3] * cols)
              top    = int(detection[4] * rows)
              right  = int(detection[5] * cols)
              bottom = int(detection[6] * rows)
              boxes.append([left, top, right, bottom])

      # apply non-max suppression
      if args.nonmax:
          indices = cv2.dnn.NMSBoxes(boxes, confidences, args.thr, args.thr - 0.1)
      else:
          indices = np.array([[i] for i in range(len(boxes))])

      # go through the detections remaining
      # after nms and draw bounding box
      for i in indices:
          i = i[0]
          box = boxes[i]
          x = box[0]
          y = box[1]
          w = box[2]
          h = box[3]
          draw_bounding_box(frame, class_ids[i], confidences[i], x, y, w, h)    #For get the class and location of object detected, 

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break
    sys.stdout.flush()
    if (not args.skip):
      sys.stdout.write("Playing video %dx%d at FPS: %.1f (source fps %d)\r"%(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT),1.0/(time.time() - start),cap.get(cv2.CAP_PROP_FPS)))
    if (not args.skip):
      sys.stdout.write("Processing video %dx%d at FPS: %.1f (source fps %d)\r"%(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT),1.0/(time.time() - start),cap.get(cv2.CAP_PROP_FPS)))
sys.stdout.write("Done\n")
