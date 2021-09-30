from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import math

eps = 1e-6

def single_pixel_convolution(input, x, y, k, kernel_width, kernel_height):
    output = 0;
    for i in range(kernel_width):
      for j in range(kernel_height):
        output = output + (input[x+i, y+j] * k[i, j])
    return output

def convolution_2d(input, width, height, kernel, kernel_width, kernel_height):
    small_width = width - kernel_width + 1
    small_height = height - kernel_height + 1
    output = np.zeros((small_width, small_height))
    # for i in range(small_width):
    #   for j in range(small_height):
    #     output[i, j]=0
    for i in range(small_width):
      for j in range(small_height):
        output[i, j] = single_pixel_convolution(input,i,j,kernel,kernel_width,kernel_height)
    return output

def convolution_2d_padded(input, width, height, kernel,	kernel_width, kernel_height):
    small_width = width - kernel_width + 1
    small_height = height - kernel_height + 1
    top = int(kernel_height/2)
    left = int(kernel_width/2)
    small = np.zeros((small_width, small_height))
    small = convolution_2d(input,width,height,kernel,kernel_width,kernel_height)
    large = np.zeros((width, height))
    # for j in range(height):
    #   for i in range(width):
    #     large[i, j] = 0;
    for j in range(small_height):
      for i in range(small_width):
        large[i+left, j+top] = small[i, j]
    return large

def gaussian_discrete_2d(theta, x, y):
    g = 0
    for y_subpixel in np.arange(y - 0.5, y + 0.55, 0.1):
      for x_subpixel in np.arange(x - 0.5, x + 0.55, 0.1):
        g = g + ((1/(2*math.pi*theta*theta)) * math.pow(math.e,-(x_subpixel*x_subpixel+y_subpixel*y_subpixel)/(2*theta*theta)))
    g = g/121
    return g

def gaussian_2d(theta, size):
    kernel = np.zeros((size, size))
    for j in range(size):
      for i in range(size):
        kernel[i, j] = gaussian_discrete_2d(theta,i-(size/2),j-(size/2))
    return kernel

def gaussian(input, ks, sigma):
    width = input.shape[0]
    height = input.shape[1]
    gaussian_kernel = gaussian_2d(sigma,ks);
    output = convolution_2d_padded(input,width,height,gaussian_kernel,ks,ks);
    return output

# From https://easysavecode.com/5jIZDikh
def pascal_row(n, memo={}):
    # This returns the nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result


class OxCaptcha():
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._image = Image.new('RGB', (width, height))
        self._draw = ImageDraw.Draw(self._image)
        self._font = ImageFont.load_default()
        self._background_color = (255, 255, 255)
        self._foreground_color = (0, 0, 0)

    def background(self, color=None):
        if color is None:
            color = self._background_color
        self._draw.rectangle([(0,0), (self._width, self._height)], fill=color)

    def foreground(self, color):
        self._foreground_color = color

    def font(self, file_name, size):
        self._font = ImageFont.truetype(file_name, size=size)

    def text(self, text, x_offset, y_offset, kerning=0, color=None):
        if color is None:
            color = self._foreground_color

        x = x_offset
        y = y_offset
        for c in text:
            self._draw.text((x, y), c, color, font=self._font)
            w, h = self._font.getsize(c)
            x += w + kerning

    def distortion_elastic(self, alpha, kernel_size, sigma):
        dx_field = np.zeros((self._height, self._width))
        dy_field = np.zeros((self._height, self._width))
        source = np.zeros((self._height, self._width))
        s = np.asarray(self._image).mean(-1)

        for y in range(self._height):
            for x in range(self._width):
                dx_field[y, x] = 2 * (random.random() - 0.5)
                dy_field[y, x] = 2 * (random.random() - 0.5)
                if random.random() < 0.1:
                    dx_field[y, x] = dx_field[y, x] * 5
                if random.random() < 0.1:
                    dy_field[y, x] = dy_field[y, x] * 5
                source[y, x] = s[y, x]

        dx_field = gaussian(dx_field, kernel_size, sigma)
        dy_field = gaussian(dy_field, kernel_size, sigma)

        for y in range(self._height):
            for x in range(self._width):
                dx = dx_field[y, x] * alpha
                dy = dy_field[y, x] * alpha

                sx = x + dx
                sy = y + dy
                if (sx < 0) or (sx > self._width - 2) or (sy < 0) or (sy > self._height - 2):
                    self._image.putpixel((x, y), self._background_color)
                else:
                    sx_left = math.floor(sx)
                    sx_right = sx_left + 1
                    sx_dist = sx % 1

                    sy_top = math.floor(sy)
                    sy_bottom = sy_top + 1
                    sy_dist = sy % 1

                    top = (1. - sx_dist) * source[sy_top, sx_left] + sx_dist * source[sy_top, sx_right]
                    bottom = (1. - sx_dist) * source[sy_bottom, sx_left] + sx_dist * source[sy_bottom][sx_right]
                    target = (1. - sy_dist) * top + sy_dist * bottom
                    t = int(max(min(target, 255), 0))
                    self._image.putpixel((x, y), (t, t, t))

    def distortion_shear(self, x_phase, x_period, x_amplitude, y_phase, y_period, y_amplitude):
        for i in range(self._width):
            dst_x = i - 1
            dst_y = int(math.sin(float(x_phase + i) / (eps + float(x_period))) * x_amplitude)
            src_x = i
            src_y = 0
            src_w = 1
            src_h = self._height
            dx = dst_x - src_x
            dy = dst_y - src_y
            strip = self._image.crop((src_x, src_y, src_x+src_w, src_y+src_h))
            self._image.paste(strip, (src_x+dx, src_y+dy))
            # _img_g.copyArea(src_x, src_y, src_w, src_h, dx, dy);
            # _img_g.setColor(_bg_color)
            if dy >= 0:
                self._draw.line((i, 0, i, dy), fill=self._background_color)
            else:
                self._draw.line((i, self._height + dy, i, self._height), fill=self._background_color)
        for i in range(self._height):
            dst_x = int(math.sin(float(y_phase + i) / (eps + float(y_period))) * y_amplitude)
            dst_y = i - 1
            src_x = 0
            src_y = i
            src_w = self._width
            src_h = 1
            dx = dst_x - src_x
            dy = dst_y - src_y
            strip = self._image.crop((src_x, src_y, src_x+src_w, src_y+src_h))
            self._image.paste(strip, (src_x+dx, src_y+dy))
            # _img_g.copyArea(src_x, src_y, src_w, src_h, dx, dy);
            # _img_g.setColor(_bg_color);
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
            xys = [(random.randint(0, self._width), random.randint(0, self._height)), (random.randint(0, self._width), random.randint(0, self._height)), (random.randint(0, self._width), random.randint(0, self._height))]
            self.draw_bezier(xys, width=width, color=color)

    def noise_ellipses(self, ellipses, width, color=None):
        if color is None:
            color = self._background_color
        for i in range(ellipses):
            xy = [(random.randint(0, self._width), random.randint(0, self._height)), (random.randint(0, self._width), random.randint(0, self._height))]
            self._draw.ellipse(xy, outline=color, width=width)

    def noise_white_gaussian(self, sigma):
        s = np.asarray(self._image).mean(-1)
        for y in range(self._height):
            for x in range(self._width):
                p = float(s[y, x])
                p = int(p + sigma * random.gauss(0, 1))
                p = max(0, min(255, p));
                self._image.putpixel((x, y), (p, p, p))

    def save(self, file_name):
        self._image.save(file_name)

    def as_array(self):
        return np.asarray(self._image).mean(-1)
