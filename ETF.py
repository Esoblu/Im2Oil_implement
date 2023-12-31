import cv2
import math
import numpy as np
import torch
import torch.nn as nn


class ETF:
    def __init__(self, img, kernel_radius, iter_time, background_dir=None):

        img = img
        self.origin_shape = img.shape
        self.shape = img.shape
        self.kernel_size = kernel_radius * 2 + 1
        self.kernel_radius = kernel_radius
        self.iter_time = iter_time
        self.background_dir = background_dir

        img = cv2.copyMakeBorder(img, kernel_radius, kernel_radius, kernel_radius, kernel_radius, cv2.BORDER_REPLICATE)
        img_normal = cv2.normalize(img.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)

        x_der = cv2.Sobel(img_normal, cv2.CV_32FC1, 1, 0, ksize=5)
        y_der = cv2.Sobel(img_normal, cv2.CV_32FC1, 0, 1, ksize=5)

        x_der = torch.from_numpy(x_der) + 1e-12
        y_der = torch.from_numpy(y_der) + 1e-12

        gradient_magnitude = torch.sqrt(x_der ** 2.0 + y_der ** 2.0)
        gradient_norm = gradient_magnitude / gradient_magnitude.max()

        x_norm = x_der / (gradient_magnitude)
        y_norm = y_der / (gradient_magnitude)

        # rotate 90 degrees counter-clockwise
        self.x_norm = -y_norm
        self.y_norm = x_norm

        self.gradient_norm = gradient_norm
        self.gradient_magnitude = gradient_magnitude

    def Ws(self):
        kernels = torch.ones((*self.shape, self.kernel_size, self.kernel_size))
        # radius = central = (self.kernel_size-1)/2
        # for i in range(self.kernel_size):
        #     for j in range(self.kernel_size):
        #         if (i-central)**2+(i-central)**2 <= radius**2:
        #              self.flow_field[x][y]
        return kernels

    def Wm(self):
        kernels = torch.ones((*self.shape, self.kernel_size, self.kernel_size))

        eta = 1  # Specified in paper
        (h, w) = self.shape
        x = self.gradient_norm[self.kernel_radius:-self.kernel_radius, self.kernel_radius:-self.kernel_radius]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                y = self.gradient_norm[i:i + h, j:j + w]
                # kernels[:, :, i, j] = (1 / 2) * (1 + torch.tanh(eta * (y - x)))
                kernels[:, :, i, j] = (1 / 2) * (1 + y - x)
        return kernels

    def Wd(self):
        kernels = torch.ones((*self.shape, self.kernel_size, self.kernel_size))

        (h, w) = self.shape
        X_x = self.x_norm[self.kernel_radius:-self.kernel_radius, self.kernel_radius:-self.kernel_radius]
        X_y = self.y_norm[self.kernel_radius:-self.kernel_radius, self.kernel_radius:-self.kernel_radius]

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                Y_x = self.x_norm[i:i + h, j:j + w]
                Y_y = self.y_norm[i:i + h, j:j + w]
                kernels[:, :, i, j] = X_x * Y_x + X_y * Y_y

        return torch.abs(kernels), torch.sign(kernels)

    def forward(self):
        Ws = self.Ws()
        Wm = self.Wm()
        x_norm = None
        y_norm = None
        for iter_time in range(self.iter_time):
            Wd, phi = self.Wd()
            kernels = phi * Ws * Wm * Wd

            x_magnitude = (self.gradient_norm * self.x_norm).unsqueeze(0).unsqueeze(0)
            # print(x_magnitude.min())
            y_magnitude = (self.gradient_norm * self.y_norm).unsqueeze(0).unsqueeze(0)

            x_patch = torch.nn.functional.unfold(x_magnitude, (self.kernel_size, self.kernel_size))
            y_patch = torch.nn.functional.unfold(y_magnitude, (self.kernel_size, self.kernel_size))

            x_patch = x_patch.view(self.kernel_size, self.kernel_size, *self.shape)
            y_patch = y_patch.view(self.kernel_size, self.kernel_size, *self.shape)

            x_patch = x_patch.permute(2, 3, 0, 1)
            y_patch = y_patch.permute(2, 3, 0, 1)

            x_result = (x_patch * kernels).sum(-1).sum(-1)
            y_result = (y_patch * kernels).sum(-1).sum(-1)

            magnitude = torch.sqrt(x_result ** 2.0 + y_result ** 2.0)
            x_norm = x_result / magnitude
            y_norm = y_result / magnitude

            self.x_norm[self.kernel_radius:-self.kernel_radius, self.kernel_radius:-self.kernel_radius] = x_norm
            self.y_norm[self.kernel_radius:-self.kernel_radius, self.kernel_radius:-self.kernel_radius] = y_norm

        angle = self.save(x_norm, y_norm)
        return angle

    def save(self, x, y):
        x = nn.functional.interpolate(x.unsqueeze(0).unsqueeze(0), [*self.origin_shape], mode='nearest')
        y = nn.functional.interpolate(y.unsqueeze(0).unsqueeze(0), [*self.origin_shape], mode='nearest')
        x = x.squeeze()
        y = y.squeeze()
        x[x == 0] += 1e-12

        tan = -y / x
        angle = torch.atan(tan)
        angle = 180 * angle / math.pi

        if self.background_dir != None:
            t = self.gradient_magnitude[self.kernel_radius:-self.kernel_radius, self.kernel_radius:-self.kernel_radius]
            t = nn.functional.interpolate(t.unsqueeze(0).unsqueeze(0), [*self.origin_shape], mode='bilinear')
            t = t.squeeze()
            # a = t.min()
            # b = t.max()
            angle[t < 0.4] = self.background_dir

        return angle
