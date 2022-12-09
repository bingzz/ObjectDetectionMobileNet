import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
from CentroidTracker import CentroidTracker
from TrackableObject import TrackableObject
from imutils.video import FPS
from imutils.video import VideoStream

config_file = "Models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "Models/frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(config_file, frozen_model)  # process
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

class_labels = []  # empty list models for identifications
file_name = "Models/Labels.txt"
thresh = 0.575
nms_thresh = 0.3
font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

count = 0  # frame count in the video
frame_skip = 1  # skip frames
counter_top = 0  # people passing to the top
counter_bottom = 0  # people passing to the bottom
offset = 5  # buffer zone

# intantiate the centroid trackers that would store a list of trackers and map a unique object ID
centroid_tracker = CentroidTracker(40, 50)
trackers = []
trackableObjects = {}

# Video statistics
frame_rate = 0
total_frames = 0

video_file = "Videos/Sample2.mp4"  # video file

with open(file_name, "rt") as fpt:
    class_labels = fpt.read().rstrip("\n").split("\n")

vid = cv2.VideoCapture(video_file)  # video
if not vid.isOpened():
    vid = cv2.VideoCapture
if not vid.isOpened():
    raise IOError("Cannot Open Video")

# fps = FPS().start() # start frame count
# frame_rate = cv2.CAP_PROP_FPS # frame rates

# PLAY VIDEO
while True:
    ret, frame = vid.read()  # read video file

    # stop video if there are no more frames
    if frame is None:
        break

    H, W = frame.shape[:2]  # get video resolution
    frame = cv2.resize(frame, (W, H))  # set video resolution

    # FAST FORWARD (FRAME SKIP)
    count += 1
    if count % frame_skip == 0:  # skip frames per second
        count = 0
    else:
        continue

    # initialize the current status and instances
    class_index, confidence, bbox = model.detect(frame, thresh)
    rectangles = []

    # bbox = list(bbox)
    # confidence = list(np.array(confidence).reshape(1,-1)[0])
    # confidence = list(map(float, confidence))

    # indices = cv2.dnn.NMSBoxes(bbox, confidence, thresh, nms_thresh)

    # for i in indices:
    #     # i = i[0]
    #     box = bbox[i]
    #     X,Y,W,H = box[0], box[1], box[2], box[3]
    #     cv2.rectangle(frame, box[0], box[1], (0,0,255), 2)
    #     cv2.putText(frame, class_labels[class_index[0]-1].upper(), (box[0]+10, box[1]+30), font, font_scale, (0,255,127), 2)

    # Overlays
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 2)  # Line in the video
    cv2.putText(frame, "Top: %s" % str(counter_top), (30, 40), font, font_scale, (127, 255, 127), 2) # top counter
    cv2.putText(frame, "Bottom: %s" % str(counter_bottom), (30, H - 40), font, font_scale, (127, 0, 127), 2)  # bottom counter

    trackers = []
    rects = []

    if len(class_index) != 0:  # detects for objects
        for class_ind, conf, boxes, person in zip(class_index.flatten(), confidence.flatten(), bbox, range(len(class_index))):  # DETECTION PROCESSING
            if class_ind == 1:  # detects a person
                center_x = boxes[0] + boxes[2] // 2
                center_y = boxes[1] + boxes[3] // 2
                centroid = (center_x, center_y)

                track_person = TrackableObject(person, centroid)

                print("ID: %s, Center: %s" % (track_person.objectID, track_person.centroid))

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(boxes[0]), int(boxes[1]), int(boxes[0]) + int(boxes[2]), int(boxes[1]) + int(boxes[3]))
                tracker.start_track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rect)

                trackers.append(tracker)

                objects = centroid_tracker.update(rects)
                
                # for (objectID, centroid) in objects.items():
                #     print(objectID)
                # CREATE A CLASS THAT WOULD HOLD THE PERSON ID AND HOLD ITS POSITION AND PLACE OF ORIGIN
                # IF PERSON ID ALREADY EXISTS, DO NOT CREATE
                # OTHERWISE, CREATE A NEW PERSON ID
                # OTHER CASES WILL BE THE TRACKER DISAPPEARS AND RE APPEARS ON THE SAME PERSON

                cv2.rectangle(frame, boxes, (0, 0, 255), 2)  # person container
                cv2.circle(frame, centroid, 5, (255, 255, 255), -1)  # dot in the middle
                # cv2.putText(frame, class_labels[class_ind - 1], (boxes[0] + 10, boxes[1]  + 20), font, font_scale,
                #             (0, 255, 127), 2)  # label
                cv2.putText(frame, "ID: %s" % (person), (boxes[0] + 10, boxes[1] + 20), font, font_scale,
                            (0, 255, 127), 2) # (left, top, width, height)

                # LINE COUNT
                # FROM TOP TO BOTTOM
                # if (H + offset) // 2 > center_y > (H - offset) // 2:
                #     if not track_person.counted:
                #         track_person.counted = True
                #         counter_bottom += 1
                #         cv2.line(frame, (0, H // 2), (W, H // 2), (0, 200, 0), 2)  # Line in the video
                # FROM BOTTOM TO TOP

    cv2.imshow("Object Detection", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

# read image
# img = cv2.imread("man-car.jpg") # image format

# plt.imshow(img) # BGR format

# class_index, confidence, bbox = model.detect(img, 0.5)
# # print(class_index)
#
# font_scale = 3
# font = cv2.FONT_HERSHEY_PLAIN
#
# for class_ind, conf, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
#     cv2.rectangle(img, boxes, (255,0,0), 2)
#     cv2.putText(img, class_labels[class_ind-1], (boxes[0]+10, boxes[1]+40), font, font_scale, (0,255,0), 3)
#
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # RGB format
# plt.show()
