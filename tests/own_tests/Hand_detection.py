# -*- coding:utf8 -*-
import cv2
import joblib
import numpy as np
import os
from time import time
import PyOpenPose as OP
from modules.Gestures import GestureRec
from modules.Hands import HandDetector
from modules.WebCamera import VideoStream


knn = joblib.load('/home/user/PycharmProjects/Hands and pose gestures recognition/classifiers/hand_gestures_classifier/knn_gesture')
gestures_combination = joblib.load('/home/user/PycharmProjects/Hands and pose gestures recognition/classifiers/hand_gestures_classifier/gestures_combination_dict')
gestures_list = []
gestures_dict = {1.: 'palm', 0.0: 'rock', 2.: 'fist', 3.: '1', 4. : '2'}

# Camera init
width, height = 640, 360
stream = VideoStream(width=width, height=height, camera_num=0)
RUN = True
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

detector = HandDetector()
gest = GestureRec()
op_root = "/home/user/openpose"
op = OP.OpenPose(
                                (640, 320), (240, 240), (width, height),"COCO",
                                op_root + "/models/" , 0, False,
                                OP.OpenPose.ScaleMode.ZeroToOne,
                                False, True
                                )

TF = True
OP = False
k_points = []
op_box = None
hand = None

while RUN:
    gesture = None
    t = time()
    ret, img = stream.get_img()
    if ret:
        if TF:
            # Looking for hands
            actual_boxes = detector.detect_hands(img, im_widh=width,
                                                                            im_height=height)
            if len(actual_boxes) > 0:
                box = actual_boxes[0]
                op_box = detector.tf_box_to_op_box(box, padding=1.5)
                x, y, dx, dy = op_box
                cv2.rectangle(img, (x, y), (x + dx, y + dy), (0, 0, 255), 5)
                array = np.array(op_box + op_box, dtype=np.int32)
                array = array.reshape((1, 8))
                op.detectHands(img, array)
                left_hand = op.getKeypoints(op.KeypointType.HAND)[0]
                right_hand = op.getKeypoints(op.KeypointType.HAND)[1]
                left_hand = left_hand.reshape(-1, 3)
                right_hand = right_hand.reshape(-1, 3)
                score, hand = gest.what_hand(left_hand, right_hand)

                if score > 0.50:
                    k_points = left_hand if hand == 'Left' else right_hand
                    op_box = gest.compute_BB(k_points)[1]
                    TF = not TF

        else:
            # If TF found hand turn on OpenPose NN and turn off TF
            if hand == 'Right':
                right_box = np.array([0, 0, 0, 0] + op_box, dtype=np.int32)
                right_box = right_box.reshape((1, 8))
                op.detectHands(img, right_box)
                right_hand = op.getKeypoints(op.KeypointType.HAND)[1]
                right_hand = right_hand.reshape(-1, 3)
                k_points = gest.right_hand_skeleton(right_hand)
                img = op.render(img)
            elif hand == 'Left':
                left_box = np.array(op_box + [0, 0, 0, 0], dtype=np.int32)
                left_box = left_box.reshape((1, 8))
                op.detectHands(img, left_box)
                left_hand = op.getKeypoints(op.KeypointType.HAND)[0]
                left_hand = left_hand.reshape(-1, 3)
                k_points = gest.left_hand_skeleton(left_hand)
                img = op.render(img)
            if len(k_points) > 0:
                op_box = gest.compute_BB(k_points)[1]
                x, y, d1, d2 =op_box
                cv2.rectangle(img, (x, y), (x + d1, y + d2), (0, 255, 255), 5)
                distance = gest.compute_distanse(k_points)
                # Gesture classification
                gesture = knn.predict(distance)[0]
                if len(gestures_list) > 24:
                    gestures_list = gestures_list[1:]
                gestures_list.append(gesture)
                gestures_list = gest.gestures_consistently(gestures_combination,
                                                           gestures_list)
            else:
                TF = not TF

    # Calculate FPS
    t = time() - t
    fps = 1.0 / t
    # Visualization
    cv2.putText(img, 'FPS = %f' % fps, (20, 20),
                        0, 0.5, (0, 0, 255), thickness=4)
    if gesture or gesture == 0.0:
        cv2.putText(img, 'Gesture = %s' % gestures_dict[gesture], (20, 55),
                    0, 1, (255, 0, 0), thickness=6)
    cv2.imshow('Video', img)
    if cv2.waitKey(10) & 0xFF == 27:
        RUN = not RUN

cv2.destroyAllWindows()
stream.stop()