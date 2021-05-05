#%%
import cv2
import numpy as np
import argparse
import time

# %%
def load_yolo():
    net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
    classes = []
    with open("./coco.names", 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        #print(classes)
    layers_name = net.getLayerNames()
    #print(layers_name)
    output_layers = [layers_name[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(output_layers)
    #print(len(classes))
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    #print(colors.shape)
    return net, classes, colors, output_layers


# %%
def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

# %%
def pre_processing(img, net, output_layers):
    #mean_subtraction, scale_factor->standard scale factor used for this version of yolo, no mean_subtraction, opencv is BGR and mean_dub is RGB so this is used
    blob = cv2.dnn.blobFromImage(img, scalefactor = 0.00392, size = (320,320), mean = (0,0,0), swapRB = True, crop = False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return blob, outputs


# %%
def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if(conf>0.5):#threshold of 50%
                center_x = int(detect[0]*width)
                center_y = int(detect[1]*height)
                w = int(detect[2]*width)
                h = int(detect[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids



# %%
""" with non-maxsupression """
def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-5), font, 1, color, 1)
        cv2.imshow("Image", img)
""" #Without non-maxsupression
def draw_labels(boxes, confs, colors, class_ids, classes, img):
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-5), font, 1, color, 1)
    cv2.imshow("Image", img)
"""

# %%
def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	start = float(0) 
	while True:
		start = start + 2000
		_, frame = cap.read()
		cap.set(cv2.CAP_PROP_POS_MSEC,(start))
		height, width, channels = frame.shape
		blob, outputs = pre_processing(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()
# %%
start_video("./new1.mp4")


# %%
def iou(box1, box2):
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    inter_area = (yi2-yi1)*(xi2-xi1)
    box1_area = (box1[3]-box1[1])*(box1[2]-box1[0])
    box2_area = (box2[3]-box2[1])*(box2[2]-box2[0])
    union_area = box1_area+box2_area-inter_area
    iou = inter_area/union_area
 
    return iou
# %%
