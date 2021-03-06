# -*- coding:utf8 -*-
from math import sqrt
import numpy as np

class GestureRec():
    """Класс для работы с OpenPose."""

    def compute_BB(self, hand, padding=1.8):
        # Расчет области поиска руки для скелетизации
        max_x = np.min(hand[:, 0])
        min_y = np.min(hand[:, 1])

        maxX = np.max(hand[:, 0])
        maxY = np.max(hand[:, 1])

        width = maxX - max_x
        height = maxY - min_y

        cx = max_x + width / 2
        cy = min_y + height / 2

        width = height = max(width, height) * padding

        max_x = cx - width / 2
        min_y = cy - height / 2

        score = np.mean(hand[:, 2])

        if max_x > 10:
            max_x -= 10
        else:
            max_x = 0

        if min_y > 10:
            min_y -= 10
        else:
            min_y = 0
        return score, [int(max_x), int(min_y), int(width) + 20, int(height) + 20]

    def what_hand(self, left_hand, righ_hand):
        # Определяем, с какой рукой будем работать

        #self.op.detectHands(img, np.array(box + box, dtype=np.int32).reshape((1, 8)))
        # leftHand = self.op.getKeypoints(self.op.KeypointType.HAND)[0].reshape(-1, 3)
        # rightHand = self.op.getKeypoints(self.op.KeypointType.HAND)[1].reshape(-1, 3)
        score_l, new_hand_bb_l = self.compute_BB(left_hand)
        score_r, new_hand_bb_r = self.compute_BB(righ_hand)
        if score_l >= score_r:
            hand = 'Left'
            return score_l, hand
        else:
            hand = 'Right'
            return score_r, hand

    def left_hand_skeleton(self, hand):
        # Скелетизация левй руки
        score, new_hand_bb = self.compute_BB(hand)
        if score > 0.5:
            k_points = hand
            #rendered_img = self.op.render(img)
            return  k_points
        else:
            return []

    def right_hand_skeleton(self, hand):
        # Скелетизация правой руки
        score, new_hand_bb = self.compute_BB(hand)
        if score > 0.5:
            k_points = hand
            return k_points
        else:
            return []

    def gesture_classification(self, k_points, classifier):
        # Классификация жестов
        if len(k_points):
            distace = self.compute_distanse(k_points)
            gesture = classifier.predict(distace)[0]
            labels = {0: 'rock', 1: 'palm', 2: 'fist', 3: '1 finger', 4: 'i fingers'}
            return labels[gesture]
        else:
            return  None

    @staticmethod
    def compute_distanse(hand):
        # Считаем расстояния от клчевых точек руки до начала ладони
        x = []
        y = []
        width = []
        for k in hand:
            x.append(k[0])
            y.append(k[1])
        for i in range(1, 21):
            width.append(sqrt((x[i] - x[0]) ** 2 + (y[i] - y[0]) ** 2))
        width = np.array(width)
        return width.reshape(1, 20)

    @staticmethod
    def compute_distanse20(hand):
        # Считаем расстояния от клчевых точек руки до начала ладони
        # И расстояние между кончиками пальцев
        x = []
        y = []
        width = []
        for k in hand:
            x.append(k[0])
            y.append(k[1])
        for i in range(1, 21):
            width.append(sqrt((x[i] - x[0]) ** 2 + (y[i] - y[0]) ** 2))
        width.append(sqrt((x[8] - x[4]) ** 2 + (y[8] - y[4]) ** 2))
        width.append(sqrt((x[12] - x[8]) ** 2 + (y[12] - y[8]) ** 2))
        width.append(sqrt((x[16] - x[12]) ** 2 + (y[16] - y[12]) ** 2))
        width.append(sqrt((x[20] - x[16]) ** 2 + (y[20] - y[16]) ** 2))
        width = np.array(width)
        return width.reshape(1, 24)


    def gestures_consistently(self, gest_dict, gest_lst):
        #  Определяем, была ли заранее продемонстрирована
        #  последовательность жестов.
        #  gest_dict -  словарь жестовых последовательностей
        #  gest_list - обновляемый список фиксируемых камерой
        #  зарезервированных жестов.

        for combination in gest_dict:
            if len(combination) <= len(gest_lst):
                found = True
                g = gest_lst[:]
                for comb_num in range(len(combination)):
                    if combination[comb_num] in g:
                        g = g[g.index(combination[comb_num]):]
                    else:
                        found = False
                if found:
                    print(gest_dict[combination])
                    gest_lst = []
                    return gest_lst
        return  gest_lst

if __name__ == '__main__':
    description = "Module  Gestures realize OpenPose data processing."
    print(description)