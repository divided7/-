# -*- coding: utf-8 -*-
"""
@Time ： 2022/9/29 16:19
@Auth ： 除以七  ➗7️⃣
@File ：批量贴图工具.py
@E-mail ：divided.by.07@gmail.com
@Github ：https://github.com/divided-by-7
@ annotation ：务必注意openCV库不支持中文路径
"""
import os
import random

import cv2
import numpy as np

# _______________________________________________________________________
# argparse:
background_dir = "random cat"  # 背景图像目录
prospect_dir = "Company Face images"  # 前景图像目录
result_dir = "save"  # 保存目录
max_scale_rate = 0.1  # 每个前景的尺寸长和宽占背景图长宽的最大比例
min_scale_rate = 0.01  # 每个前景的尺寸长和宽占背景图长宽的最小比例
prospect_per_background = 20  # 每个背景图片上有多少个前景图
visual, fps = True, 5  # 可视化，若为True则fps为图片刷新速度，单位ms
random_perspective = True  # 随机透视
# _______________________________________________________________________

background_img_dir = os.listdir(background_dir)
prospect_img_dir = os.listdir(prospect_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

for idx, prospect in enumerate(prospect_img_dir):
    if idx % prospect_per_background == 0:
        if idx // prospect_per_background < len(background_img_dir):
            b_idx = idx // prospect_per_background
        else:
            b_idx = len(background_img_dir) - 1
        b = cv2.imread(background_dir + "/" + background_img_dir[b_idx])
    p = cv2.imread(prospect_dir + "/" + prospect)
    if visual:
        cv2.imshow("b", b)
        cv2.waitKey(fps)
        # cv2.destroyAllWindows()

    if max(p.shape[0] / b.shape[0], p.shape[1] / b.shape[1]) > max_scale_rate:
        rate = max(p.shape[0] / b.shape[0], p.shape[1] / b.shape[1]) / max_scale_rate
        p = cv2.resize(p, (int(p.shape[0] / rate), int(p.shape[1] / rate)))

    h, w = p.shape[:2]
    global_x0, global_y0 = int(random.uniform(0, b.shape[1] - w)), int(random.uniform(0, b.shape[0] - h))
    print("idx=", idx, ":", global_x0, global_y0)

    if random_perspective:
        if random.uniform(0, 1) > 0.5:
            crop_h, crop_w = p.shape[0], p.shape[1]  # crop_h,crop_w = 720, 1280
            # print(crop_h, crop_w)
            pts1 = np.float32([[0, 0], [crop_w, 0], [0, crop_h], [crop_w, crop_h]])
            random_perspective_rate = 0.34
            x1, y1 = random.randint(0, int(random_perspective_rate * crop_w)), random.randint(0,
                                                                                              int(random_perspective_rate * crop_h))
            x2, y2 = random.randint(int((1 - random_perspective_rate) * crop_w), crop_w), random.randint(0,
                                                                                                         int(0.34 * crop_h))
            x3, y3 = random.randint(0, int(random_perspective_rate * crop_w)), random.randint(
                int((1 - random_perspective_rate) * crop_h), crop_h)
            x4, y4 = random.randint(int((1 - random_perspective_rate) * crop_w), crop_w), random.randint(
                int((1 - random_perspective_rate) * crop_h), crop_h)
            pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            # pts2 = np.float32([[0,0],[crop_w,0],[0,crop_h],[crop_w,crop_h]])
            # print([x1, y1], [x2, y2], [x3, y3], [x4, y4])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            p = cv2.warpPerspective(p, M, (crop_w, crop_h), borderValue=(255, 255, 255))
    b[global_y0:h + global_y0, global_x0:w + global_x0] = p
    # print(idx - prospect_per_background - 1,prospect_per_background - 1)
    if (idx - prospect_per_background + 1) % (prospect_per_background) == 0:
        cv2.imwrite(result_dir + "/" + str(idx) + "cat_img.jpg", b)
        print("已保存图片" + result_dir + "/" + str(idx) + "cat_img.jpg")
