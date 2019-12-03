# import required packages
import cv2
import argparse
import numpy as np
import time

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-m', '--model', required=False, nargs='?', const=1, type=str, default='SSDv2',
                help = 'pre-trained model - allowed values:  [SSDv1q, SSDv2q, SSDv2, SSDv3]')
ap.add_argument('-O', '--nonmax', required=False, action='store_true', default=False,
                help = 'flag to enble non-max suppression (true)')
ap.add_argument('-o', '--output', required=False,
                help = 'path to text file containing class names')
ap.add_argument('-c', '--confthresh', required=False, type=float, default=0.5,
                help = 'confidence threshold')
args = ap.parse_args()

# read model
# find tensorflow models at: 
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo
if args.model=='SSDv2':
    pb = 'models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
    pbt = 'models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
elif args.model=='SSDv1q':
    pb= 'models/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/tflite_graph.pb'
    pbt='models/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/tflite_graph.pbtxt'
elif args.model=='SSDv2q':
    pb= 'models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pb'
    pbt='models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pbtxt'
elif args.model=='SSDv3':
    pb= 'models/ssd_mobilenet_v3_small_coco_2019_08_14/frozen_inference_graph.pb'
    pbt='models/ssd_mobilenet_v3_small_coco_2019_08_14/model.tflite'

# function fro drawing boxes around detencted objects
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classNames[class_id]) + ": %.2f"%confidence
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# read input image
frame = cv2.imread(args.image)
rows, cols, channels = frame.shape

# read class names from text file
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

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))

start = time.time() 
# read pre-trained model and config file
#net = cv2.dnn.readNetFromTensorflow(pb,pbt)
# create input blob 
#blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)

classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

# set input blob for the network
net.setInput(blob)
print("Model loading: %f"%(time.time() - start))

# run inference through the network
# and gather predictions from output layers
start = time.time() 
outs = net.forward()
print("Detection: %f"%(time.time() - start))

start = time.time() 
# initialization
class_ids = []
confidences = []
boxes = []

# loop over all detected objects
for detection in outs[0, 0, :, :]:        
    # get the score
    print(detection)
    score = float(detection[2])
    # draw the bounding box
    if score > args.confthresh: 
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
    print("Non-max suppression...")
    indices = cv2.dnn.NMSBoxes(boxes, confidences, args.confthresh, args.confthresh - 0.1)
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
    draw_bounding_box(frame, class_ids[i], confidences[i], x, y, w, h)

print("Post-Processing: %f"%(time.time() - start))
start = time.time() 
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow("frame", frame)
print("Drawing: %f"%(time.time() - start))
cv2.waitKey(0)
cv2.destroyAllWindows()
