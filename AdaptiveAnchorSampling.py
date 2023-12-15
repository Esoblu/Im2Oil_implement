import cv2
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans


class adaptiveSampling:
    def __init__(self, gray_img, p_max, cluster_iter=2):
        self.gray_img = gray_img
        self.p_max = p_max
        self.w = gray_img.shape[1]
        self.h = gray_img.shape[0]
        print(self.w, self.h)
        self.normalized = None
        self.K = None
        self.cluster_iter = cluster_iter
        self.sample_points = []

    def genProbabilityDensityMap(self):
        p_min = self.p_max/100   # use p_min = 0.01*p_max so that p_max is the only hyperparameter

        sobel_x = cv.Sobel(self.gray_img, cv.CV_32F, 1, 0, ksize=5)
        sobel_y = cv.Sobel(self.gray_img, cv.CV_32F, 0, 1, ksize=5)
        gradient = cv.magnitude(sobel_x, sobel_y)

        # gradient_angle = cv.phase(sobel_x, sobel_y, angleInDegrees=False)   # from 0 to 2*pi
        mean = cv.blur(gradient, (5, 5))

        self.normalized = cv2.normalize(mean, None, p_min, self.p_max, cv.NORM_MINMAX)
        self.K = np.sum(self.normalized).astype(np.int32)

    def rejectionSampling(self):
        I_u = np.random.uniform(0, 1, self.normalized.shape)
        # print(I_u.shape)
        random_series = np.random.permutation(self.w*self.h)
        i = 0
        print("K:", self.K)
        while len(self.sample_points) < self.K and i < len(random_series):
            r = random_series[i]
            i += 1
            x = r % self.w
            y = r // self.w
            if I_u[y, x] <= self.normalized[y, x]:
                self.sample_points.append([y, x])
        self.K = len(self.sample_points)
        print(self.K, "anchors")

    def kmeansClustering(self):
        kmeans = KMeans(n_clusters=self.K, init=self.sample_points, n_init=1, max_iter=self.cluster_iter)
        points = []
        for x in range(self.w):
            for y in range(self.h):
                points.append([y, x])
        kmeans.fit(points)
        return kmeans.cluster_centers_

    def get_adaptive_sampling_anchors(self):
        self.genProbabilityDensityMap()
        self.rejectionSampling()
        return self.normalized, self.kmeansClustering()


def showDensityMap(normalizedMap):
    cv.imshow('Density map', normalizedMap*100.0)
    cv.waitKey(0)
    cv.destroyAllWindows()


# if __name__ == '__main__':
#     np.random.seed(0)
#     img = cv.imread('./input/Bird.jpg')
#     p_max = 1/9
#     resized_img = img.copy()
#     if resized_img.shape[0] > resized_img.shape[1]:
#         resized_img = cv.resize(resized_img, (500, int(500 * resized_img.shape[1] / resized_img.shape[0])),
#                                 interpolation=cv.INTER_CUBIC)
#     elif resized_img.shape[0] < resized_img.shape[1]:
#         resized_img = cv.resize(resized_img, (int(500 * resized_img.shape[0] / resized_img.shape[1]), 500),
#                                 interpolation=cv.INTER_CUBIC)
#     else:
#         resized_img = cv.resize(resized_img, (500, 500), interpolation=cv.INTER_CUBIC)

#     gray_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
#     HSV_img = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)

#     density_map, K, gradient, gradient_angle = genProbabilityDensityMap(gray_img, p_max)
#     print(K)

#     sample_points = rejectionSampling(K, density_map)
#     print(len(sample_points))
#     K = len(sample_points)
#     empty_img = np.zeros(gray_img.shape, dtype=np.uint8)
#     for (x, y) in sample_points:
#         cv.circle(empty_img, (x, y), 1, (255, 0, 0), -1)

#     kmeans = KMeans(n_clusters=K, init=sample_points, n_init=1, max_iter=2)
#     points = []
#     for x in range(density_map.shape[1]):
#         for y in range(density_map.shape[0]):
#             points.append([x, y])
#     kmeans.fit(points)
#     new_sample = kmeans.cluster_centers_
#     empty_img1 = np.zeros(gray_img.shape, dtype=np.uint8)
#     for t in new_sample:
#         # print(t)
#         cv.circle(empty_img1, (int(t[0]), int(t[1])), 1, (255, 0, 0), -1)
#     cv.imshow('ori', empty_img)
#     cv.imshow('new', empty_img1)
#     cv.waitKey(0)
#     cv.destroyAllWindows()