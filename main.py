import cv2 as cv
import numpy as np
from AdaptiveAnchorSampling import adaptiveSampling
from PrintProcess import Paint
import ETF
import time
import torch
import voronoi
import tqdm


def save_anchor_points(h, w, sample_points):
    img = np.zeros((h, w), dtype=np.uint8)+255

    for sample_point in sample_points:
        y, x = sample_point
        cv.circle(img, (x, y), 1, (0, 0, 0), -1)
    cv.imwrite('./output/anchor3.png', img)


if __name__ == '__main__':
    # hyperparameters
    p_max = 1 / 25
    t_h = 60
    t_v = 15
    cluster_iter = 3
    enlarge_rate = 6
    process_size = 500
    np.random.seed(0)

    # read image
    pic_name = 'A8'
    img = cv.imread('./input/' + pic_name + '.jpg')
    print(img.shape)
    brush = cv.imread('./brush/02.png')

    if img.shape[0] < img.shape[1]: # h < w
        resized_img = cv.resize(img, (process_size, int(process_size*img.shape[0]/img.shape[1])),
                                interpolation=cv.INTER_CUBIC)
    else:
        resized_img = cv.resize(img, (int(process_size*img.shape[1]/img.shape[0]), process_size),
                                interpolation=cv.INTER_CUBIC)

    gray_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
    HSV_img = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)
    total_start = time.time()

    print('Adaptive Anchor Sampling')
    start = time.time()
    adaptive_sampler = adaptiveSampling(gray_img, p_max, cluster_iter)
    density_map, anchor_points = adaptive_sampler.get_adaptive_sampling_anchors()
    # anchor_points = np.round(anchor_points).astype(np.int32)
    anchor_points = np.int32(anchor_points)
    print("real anchor number: ", len(anchor_points))
    # print(np.max(anchor_points, axis=0), np.max(anchor_points, axis=1))
    print('Adaptive Anchor Sampling Time cost: ', time.time() - start)
    # show_sample_points(density_map.shape[1], density_map.shape[0], sample_points)
    save_anchor_points(density_map.shape[0], density_map.shape[1], anchor_points)


    print("ETF start")
    start = time.time()
    etf = ETF.ETF(gray_img, 2, 30)
    # gradient_angle = genETFVectorField(gradient, gradient_angle, new_sample_points)
    angle_hatch = np.deg2rad(etf.forward().numpy())
    angle_hatch = np.mod(angle_hatch + 2*np.pi, 2*np.pi)
    print("ETF Time cost: ", time.time() - start)

    print("Paint start")
    start = time.time()
    paint = Paint(brush, density_map, anchor_points, angle_hatch, HSV_img, p_max, t_h, t_v, enlarge_rate, pic_name)
    paint.overallPaintProcess()
    print('Paint Time cost: ', time.time() - start)
    print('Total Time cost: ', time.time() - total_start)
