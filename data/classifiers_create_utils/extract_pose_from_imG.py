import cv2
import os
import pandas as pd
import PyOpenPose as OP
import numpy as np

OPENPOSE_ROOT = "/home/user/openpose"

op = OP.OpenPose((640, 320), (240, 240), (640, 360), "COCO",
                 OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                 False,  OP.OpenPose.ScaleMode.ZeroToOne,
                 False, False)
zz = np.zeros((360, 640, 3))
for dir in os.listdir('.'):
    if os.path.isdir(dir):
        if len(os.listdir(dir)) > 10:
            np_path = './np_arrays/'+dir
            if not  os.path.exists(np_path):
                os.mkdir(np_path)
            count = 0
            for img in os.listdir(dir):
                cv_img = cv2.imread('./'+dir+'/' + img)
                z = zz.copy()
                op.detectPose(cv_img)
                img = op.render(cv_img)
                #print(img.shape)

                persons = op.getKeypoints(op.KeypointType.POSE)[0]
                if type(persons) != type(None):
                    #print(persons)
                    count = 0
                    for p in persons[0]:
                        # print(p[0], p[1])
                        if count in (2, 5):
                            cv2.circle(z, (int(p[0]) // 2, int(p[1]) // 2), 2, (0, 0, 255), thickness=4)
                        elif count in (3, 4):
                            cv2.circle(z, (int(p[0]) // 2, int(p[1]) // 2), 2, (0, 255, 0), thickness=4)
                        elif count in (6, 7):
                            cv2.circle(z, (int(p[0]) // 2, int(p[1]) // 2), 2, (255, 0, 0), thickness=4)
                        elif count in (8, 11):
                            cv2.circle(z, (int(p[0]) // 2, int(p[1]) // 2), 2, (255, 255, 0), thickness=4)
                        else:
                            cv2.circle(z, (int(p[0])//2, int(p[1])//2), 2, (255, 255, 255), thickness=4)
                        count += 1
                    cv2.imshow('img', img)
                    cv2.imshow('z', z)
                    cv2.waitKey()

                    # for person in persons:
                    #     np_arr = np.array(person)
                    #     f_name = np_path+'/'+str(img)+'.npy'
                    #     np.save(f_name, np_arr)
                    #     print(f_name)
