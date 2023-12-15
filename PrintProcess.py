import numpy as np
import cv2 as cv
from PIL import Image
import time
import os

def save_anchor_points(h, w, sample_points):
    img = np.zeros((h, w), dtype=np.uint8)
    for sample_point in sample_points:
        x, y = sample_point
        cv.circle(img, (x*6, y*6), 2, 255, -1)
    cv.imwrite('./output/anchor.png', img)


class Paint:
    def __init__(self, brush, density_map, anchor_points, angle_hatch, HSV_img, p_max, t_h, t_v, enlarge_rate,
                 pic_name):
        self.h, self.w = density_map.shape
        self.brush = brush
        self.density_map = density_map
        self.anchor_points = anchor_points
        self.angle_hatch = angle_hatch
        self.HSV_img = HSV_img
        self.p_max = p_max
        self.t_h = t_h
        self.t_v = t_v
        self.enlarge_rate = enlarge_rate
        gray_brush = cv.cvtColor(brush, cv.COLOR_BGR2GRAY)
        # gray_brush = gray_brush[np.where(gray_brush > 0)]
        gm = np.mean(gray_brush)
        print(gm)
        self.gm = gm
        self.pic_name = pic_name
        self.rst_img = None
        self.to_paint = np.zeros((len(self.anchor_points), 7), dtype=np.float32)
        self.padding_point = None

    def paintPattern(self, img, resized_brush, x, y, alpha):
        resized_brush = Image.fromarray(resized_brush)
        brush_rotated = resized_brush.rotate(np.rad2deg(alpha), expand=True)

        x -= brush_rotated.size[0] / 2.0
        y -= brush_rotated.size[1] / 2.0

        mask = brush_rotated.convert('L')
        # mask = np.array(brush_rotated.convert('L'))
        # _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
        # mask = Image.fromarray(mask)
        img.paste(brush_rotated, (int(x), int(y)), mask)

    def genBrush(self, l, w, x0, y0):
        x0, y0 = int(x0), int(y0)
        brush = cv.cvtColor(self.brush, cv.COLOR_BGR2HSV)
        result1 = self.HSV_img[y0, x0, 0].astype(np.uint16) + brush[:, :, 0].astype(np.uint16) - brush[150, 180, 0].astype(np.uint16)
        brush[:, :, 0] = np.clip(result1, 0, 255).astype(np.uint8)
        result2 = self.HSV_img[y0, x0, 1].astype(np.uint16) + brush[:, :, 1].astype(np.uint16) - brush[150, 180, 1].astype(np.uint16)
        brush[:, :, 1] = np.clip(result2, 0, 255).astype(np.uint8)
        result3 = brush[:, :, 2].astype(np.float32) * self.HSV_img[y0, x0, 2].astype(np.float32) / self.gm
        brush[:, :, 2] = np.clip(result3, 0, 255).astype(np.uint8)
        brush = cv.cvtColor(brush, cv.COLOR_HSV2RGB)
        brush = cv.resize(brush, (int(l), int(w)), interpolation=cv.INTER_CUBIC)
        # cv.imwrite('./output/brush.png', brush)
        return brush

    def paintPatterns(self):
        for i in range(len(self.to_paint) - 1, -1, -1):
            l, w, x, y, alpha, x0, y0 = self.to_paint[i]
            brush = self.genBrush(l, w, x0, y0)
            self.paintPattern(self.rst_img, brush, x, y, alpha)

    def padBackground(self):
        back_img = Image.new('RGB', (self.w * self.enlarge_rate, self.h * self.enlarge_rate), (0, 0, 0))
        for i in range(len(self.padding_point) - 1, -1, -1):
            l, w, x, y, alpha, x0, y0 = self.padding_point[i]
            brush = self.genBrush(l, w, x0, y0)
            self.paintPattern(back_img, brush, x, y, alpha)
        # back_img.save('./output/pad1.png')
        mask = np.array(self.rst_img.convert('L'))
        _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
        mask = Image.fromarray(mask)
        back_img.paste(self.rst_img, (0, 0), mask)
        self.rst_img = back_img

    def initPaint(self):
        for i in range(len(self.anchor_points)):
            y0, x0 = self.anchor_points[i]
            p = self.density_map[y0, x0]
            alpha = self.angle_hatch[y0, x0]

            l1 = self.searchLength(x0, y0, alpha, p, True)
            l2 = self.searchLength(x0, y0, alpha + np.pi, p, True)
            w1 = self.searchLength(x0, y0, alpha + np.pi / 2, p, False)
            w2 = self.searchLength(x0, y0, alpha + 3 * np.pi / 2, p, False)

            x = (x0 + ((l1 - l2) * np.cos(alpha) + (w2 - w1) * np.sin(alpha)) / 2) * self.enlarge_rate
            y = (y0 + ((w1 - w2) * np.cos(alpha) + (l1 - l2) * np.sin(alpha)) / 2) * self.enlarge_rate

            self.to_paint[i] = [(l1 + l2) * self.enlarge_rate, (w1 + w2) * self.enlarge_rate, x, y, alpha, x0, y0]
        self.to_paint = sorted(self.to_paint, key=lambda x: x[0] * x[1])

    def searchLength(self, x0, y0, alpha, p, LorW):
        alpha = np.mod(alpha, 2 * np.pi)
        if LorW:
            L_min = 3 * np.power(self.p_max, -0.5)
            L_max = 3 * np.power(p, -0.5)
        else:
            L_min = np.power(self.p_max, -0.5)
            L_max = np.power(p, -0.5)
        L = 0
        if 0 <= alpha < np.pi / 4 or 3 * np.pi / 4 <= alpha < 5 * np.pi / 4 or 7 * np.pi / 4 <= alpha < 2 * np.pi:
            k = np.tan(alpha)
            sqrt1plusk2 = np.sqrt(1 + k ** 2)
            if np.pi / 2 <= alpha < 3 * np.pi / 2:
                step_x = -1
            else:
                step_x = 1
            x = x0 + step_x
            y = y0 + k * step_x
            h0 = self.HSV_img[y0, x0, 0].astype(np.int16)
            v0 = self.HSV_img[y0, x0, 2].astype(np.int16)
            while 0 <= x < self.w and 0 <= y <= self.h - 1 and L <= L_max:
                y_int = int(y)
                h1 = self.HSV_img[y_int, x, 0].astype(np.int16)
                v1 = self.HSV_img[y_int, x, 2].astype(np.int16)
                if not ((abs(h1-h0) <= self.t_h or abs(h1+180-h0) <= self.t_h or abs(h1-180-h0) <= self.t_h) and abs(v1-v0) <= self.t_v):
                    return np.clip(L, L_min, L_max)

                L += sqrt1plusk2
                x += step_x
                y += k * step_x
        else:
            k = np.tan(np.pi / 2 - alpha)
            sqrt1plusk2 = np.sqrt(1 + k ** 2)
            if np.pi <= alpha < 2 * np.pi:
                step_y = -1
            else:
                step_y = 1
            y = y0 + step_y
            x = x0 + k * step_y
            h0 = self.HSV_img[y0, x0, 0].astype(np.int16)
            v0 = self.HSV_img[y0, x0, 2].astype(np.int16)
            while 0 <= x <= self.w - 1 and 0 <= y < self.h and L <= L_max:
                x_int = int(x)
                h1 = self.HSV_img[y, x_int, 0].astype(np.int16)
                v1 = self.HSV_img[y, x_int, 2].astype(np.int16)
                if not ((abs(h1-h0) <= self.t_h or abs(h1+180-h0) <= self.t_h or abs(h1-180-h0) <= self.t_h) and abs(v1-v0) <= self.t_v):
                    return np.clip(L, L_min, L_max)

                L += sqrt1plusk2
                x += k * step_y
                y += step_y
        return np.clip(L, L_min, L_max)

    def searchPaddingPos(self):
        # 找到图片中所有像素值为0的区域的质心
        img = np.array(self.rst_img)
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        _, img = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
        img = cv.bitwise_not(img)
        contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        self.padding_point = []
        points = []
        for contour in contours:
            M = cv.moments(contour)
            if M['m00'] < 1e-5:
                continue
            if cv.contourArea(contour) < 10:
                continue
            cx = np.int32(M['m10'] / M['m00'] / self.enlarge_rate)
            cy = np.int32(M['m01'] / M['m00'] / self.enlarge_rate)
            points.append([cx, cy])
            if cx < 0 or cx >= self.w or cy < 0 or cy >= self.h:
                continue
            p = self.density_map[cy, cx]
            alpha = self.angle_hatch[cy, cx]
            l1 = self.searchLength(cx, cy, alpha, p, True)
            l2 = self.searchLength(cx, cy, alpha + np.pi, p, True)
            w1 = self.searchLength(cx, cy, alpha + np.pi / 2, p, False)
            w2 = self.searchLength(cx, cy, alpha + 3 * np.pi / 2, p, False)
            # print(l1, l2, w1, w2)
            x = (cx + ((l1 - l2) * np.cos(alpha) + (w2 - w1) * np.sin(alpha)) / 2) * self.enlarge_rate
            y = (cy + ((w1 - w2) * np.cos(alpha) + (l1 - l2) * np.sin(alpha)) / 2) * self.enlarge_rate
            self.padding_point.append([(l1 + l2) * self.enlarge_rate, (w1 + w2) * self.enlarge_rate, x, y, alpha, cx, cy])
        print(len(self.padding_point))
        self.padding_point.sort(key=lambda x: x[0] * x[1])
        self.padding_point = np.array(self.padding_point)
        save_anchor_points(self.h*self.enlarge_rate, self.w*self.enlarge_rate, points)

    def overallPaintProcess(self):
        path = './output/' + self.pic_name
        if not os.path.exists(path):
            os.mkdir(path)
        self.rst_img = Image.new('RGB', (self.w * self.enlarge_rate, self.h * self.enlarge_rate), (0, 0, 0))
        start = time.time()
        self.initPaint()
        self.paintPatterns()
        print("1st paint time: ", time.time() - start)
        self.rst_img.save(path + '/1.png')

        i = 2
        while True:
            start = time.time()
            self.searchPaddingPos()
            self.padBackground()
            print("{} paint time: ".format(i), time.time() - start)
            self.rst_img.save(path + '/{}.png'.format(i))
            i += 1
            #exit(0)
            if len(self.padding_point) < 50:
                break
