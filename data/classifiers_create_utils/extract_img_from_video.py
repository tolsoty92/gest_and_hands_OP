# import cv2
# import os
# import PyOpenPose as OP
# #
#
# for vid in os.listdir('./video2/'):
#     if not os.path.isdir(vid):
#         os.mkdir(vid[:-4])
#
#     count = 0
#     video = cv2.VideoCapture('./video2/'+vid)
#     video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
#
#     RUN = True
#     while (video.isOpened()) and RUN:
#         ret, img = video.read()
#         if ret:
#             cv2.imshow('img', img)
#             path = './' + vid[:-4] + '/' + str(count) + '.jpg'
#             cv2.imwrite(path, img)
#             count += 1
#             if cv2.waitKey(1) & 0xFF == 27:
#                 RUN = not RUN
#         else:
#             RUN = not RUN
#
#             cv2.destroyAllWindows()
#     video.release()
#     cv2.destroyAllWindows()
