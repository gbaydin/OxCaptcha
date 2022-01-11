from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import math
import torch
import torchvision
from functools import lru_cache

eps = 1e-6


def gaussian(input, kernel_size, sigma):
    t = torch.from_numpy(input)
    if input.ndim == 2:
        t = t.unsqueeze(0)
    t = torchvision.transforms.functional.gaussian_blur(t, kernel_size=kernel_size, sigma=sigma)
    if input.ndim == 2:
        t = t.squeeze(0)
    return t


@lru_cache(maxsize=512)
def pixel_int(p):
    return int(max(min(p, 255), 0))


@lru_cache(maxsize=32768)
def distortion_shear_y(x_phase, x_period, x_amplitude, i):
    dst_y = int(math.sin(float(x_phase + i) / (eps + float(x_period))) * x_amplitude)
    return dst_y


@lru_cache(maxsize=32768)
def distortion_shear_x(y_phase, y_period, y_amplitude, i):
    dst_x = int(math.sin(float(y_phase + i) / (eps + float(y_period))) * y_amplitude)
    return dst_x


# From https://easysavecode.com/5jIZDikh
@lru_cache(maxsize=32)
def pascal_row(n):
    # This returns the nth row of Pascal's Triangle
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n & 1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    return result


class OxCaptcha():
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._image = Image.new('L', (width, height))
        self._draw = ImageDraw.Draw(self._image)
        self._font = ImageFont.load_default()
        self._background_color = 255
        self._foreground_color = 0

    def background(self, color=None):
        if color is None:
            color = self._background_color
        self._draw.rectangle([(0, 0), (self._width, self._height)], fill=color)

    def foreground(self, color):
        self._foreground_color = color

    def font(self, file_name, size):
        self._font = ImageFont.truetype(file_name, size=size)

    @lru_cache(maxsize=64)
    def text_size(self, c):
        w, h = self._font.getsize(c)
        return w, h

    def text(self, text, x_offset, y_offset, kerning=0, color=None):
        if color is None:
            color = self._foreground_color

        x = x_offset
        y = y_offset
        for c in text:
            self._draw.text((x, y), c, color, font=self._font)
            w, h = self.text_size(c)
            x += w + kerning

    def distortion_elastic(self, alpha, kernel_size, sigma):
        source = np.asarray(self._image)

        d_field = 2 * (np.random.rand(2, self._height, self._width)-0.5)
        mask = 1 + 4.*(np.random.rand(2, self._height, self._width) < 0.1)
        d_field = d_field * mask
        d_field = gaussian(d_field, kernel_size, sigma) * alpha

        for y in range(self._height):
            for x in range(self._width):
                dx = d_field[0, y, x]
                dy = d_field[1, y, x]

                sx = x + dx
                sy = y + dy
                if (sx < 0) or (sx > self._width - 2) or (sy < 0) or (sy > self._height - 2):
                    self._image.putpixel((x, y), self._background_color)
                else:
                    sx_left = math.floor(sx)
                    sy_top = math.floor(sy)
                    target = source[sy_top, sx_left]
                    t = pixel_int(target)
                    self._image.putpixel((x, y), t)

    def distortion_shear(self, x_phase, x_period, x_amplitude, y_phase, y_period, y_amplitude):
        for i in range(self._width):
            dy = distortion_shear_y(x_phase, x_period, x_amplitude, i)
            strip = self._image.crop((i, 0, i+1, self._height))
            self._image.paste(strip, (i-1, dy))
            if dy >= 0:
                self._draw.line((i, 0, i, dy), fill=self._background_color)
            else:
                self._draw.line((i, self._height + dy, i, self._height), fill=self._background_color)
        for i in range(self._height):
            dx = distortion_shear_x(y_phase, y_period, y_amplitude, i)
            strip = self._image.crop((0, i, 0+self._width, i+1))
            self._image.paste(strip, (0+dx, i-1))
            if dx >= 0:
                self._draw.line((0, i, dx, i), fill=self._background_color)
            else:
                self._draw.line((self._width+dx, i, self._width, i))

    # From https://easysavecode.com/5jIZDikh
    def draw_bezier(self, xys, width, color):
        # xys should be a sequence of 2-tuples (Bezier control points)
        n = len(xys)
        combinations = pascal_row(n-1)

        def bezier(ts):
            # This uses the generalized formula for bezier curves
            # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
            result = []
            for t in ts:
                tpowers = (t**i for i in range(n))
                upowers = reversed([(1-t)**i for i in range(n)])
                coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
                result.append(
                    tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
            return result

        ts = [t/100.0 for t in range(101)]
        points = bezier(ts)
        for i in range(len(points)-1):
            self._draw.line([points[i], points[i+1]], fill=color, width=width)

    def noise_strokes(self, strokes, width, color=None):
        if color is None:
            color = self._foreground_color
        for i in range(strokes):
            xys = [(random.randint(0, self._width), random.randint(0, self._height)), (random.randint(0, self._width),
                                                                                       random.randint(0, self._height)), (random.randint(0, self._width), random.randint(0, self._height))]
            self.draw_bezier(xys, width=width, color=color)

    def noise_ellipses(self, ellipses, width, color=None):
        if color is None:
            color = self._background_color
        for i in range(ellipses):
            xy = [(random.randint(0, self._width), random.randint(0, self._height)), (random.randint(0, self._width), random.randint(0, self._height))]
            self._draw.ellipse(xy, outline=color, width=width)

    def noise_white_gaussian(self, sigma):
        s = np.asarray(self._image) + np.random.normal(0, sigma, (self._height, self._width))
        # self._image.putdata(s)
        for y in range(self._height):
            for x in range(self._width):
                p = s[y, x]
                p = pixel_int(p)
                self._image.putpixel((x, y), p)

    def save(self, file_name):
        self._image.save(file_name)

    def as_array(self):
        return np.asarray(self._image)
