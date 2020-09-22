from dataclasses import dataclass

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from collections import namedtuple
from random import randrange as rr
import pathlib
import random
import os
import time
import configuration
from tqdm import tqdm


@dataclass
class Star:
    x: int
    y: int
    brightness: int
    fwhm: int

    def toTSV(self):
        return [self.x, self.y, self.brightness, self.fwhm]


@dataclass
class Object:
    x: int
    y: int
    brightness: int
    fwhm: int
    positions: list

    def toTSV(self, pos):
        return [self.positions[pos][0], self.positions[pos][1], self.brightness, self.fwhm]


class StarGenerator:

    def __init__(self, config):
        self.config: configuration.Configuration = config

    def generateSeries(self):
        for i in tqdm(range(self.config.numberOfSeries)):
            self.generateOneSeries()

    def generateOneSeries(self):

        t = time.time()

        objects = [self.randomObject() for i in range(self.config.Objects.count.value())]
        stars = self.generateStars()

        self.saveTSV(stars, objects, t)

        if self.config.saveImages:

            stars_image = np.zeros((self.config.SizeX, self.config.SizeY))
            for s in stars:
                self.drawStarGaus(s, stars_image)

            images = []
            for i in range(8):
                image = stars_image.copy()
                self.addNoise(image)

                for obj in objects:
                    self.drawStarGaus(obj, image)
                    obj.x, obj.y = obj.positions[(i + 1) % 8][0], obj.positions[(i + 1) % 8][1]

                images.append(image)

            self.saveSeriesToFile(images, objects)

            if self.config.plot:
                self.plotSeries(images)

    def saveTSV(self, stars, objects, t):

        directory = os.path.join(self.config.dataFile, f'{t}')
        os.mkdir(directory)

        for i in range(8):
            data = [s.toTSV() for s in stars] + [o.toTSV(i) for o in objects]
            df = pd.DataFrame(np.array(data), columns=["x", "y", "brightness", "fwhm"])
            df.to_csv(f"{directory}/data_{i}'.csv", index=False)

        data = [[i] + o.toTSV(i) for o in objects for i in range(8)]
        df = pd.DataFrame(np.array(data), columns=["image_number", "x", "y", "brightness", "fwhm"])
        df.to_csv(f"{directory}/objects.csv", index=False)

    def plotSeries(self, images):
        fig, axs = plt.subplots(2, 4)
        for r in range(2):
            for c in range(4):
                axs[r, c].imshow(images[4 * r + c], cmap='gray', vmin=0, vmax=50)
                axs[r, c].set_title(f'image {4 * r + c}')
        plt.show()

    def saveSeriesToFile(self, images, objects):
        directory = os.path.join(self.config.dataFile, f'{int(time.time())}')
        os.mkdir(directory)
        for i in range(8):
            name = f'{directory}/{i}'
            self.saveImgToFits(images[i], name)

        with open(f'{directory}/objects.txt', 'w') as f:
            for obj in objects:
                print(' '.join(list(map(str, obj.positions))), file=f)

    def generateStars(self):

        stars = [self.randomStar() for i in range(self.config.Stars.count.value())]

        return stars

    def randomStar(self):
        x = rr(self.config.SizeX)
        y = rr(self.config.SizeY)
        brightness = self.config.Stars.brightness.value()
        fwhm = self.config.Stars.fwhm.value()

        return Star(x, y, brightness, fwhm)

    def randomObject(self):
        points = self.generateRandomObjectPoints()
        brightness = self.config.Objects.brightness.value()
        fwhm = self.config.Objects.fwhm.value()

        obj = Object(x=points[0][0], y=points[0][1],
                     brightness=brightness, fwhm=fwhm, positions=points)

        return obj

    def generateRandomObjectPoints(self):
        x = rr(self.config.SizeX)
        y = rr(self.config.SizeY)
        p = random.random()
        if p >= 0.75:
            edge_point = (rr(self.config.SizeX), 0)
        elif p >= 0.5:
            edge_point = (rr(self.config.SizeX), 1024)
        elif p >= 0.25:
            edge_point = (0, rr(self.config.SizeY))
        else:
            edge_point = (1024, rr(self.config.SizeY))
        speed = self.config.Objects.speed.value() / 100

        step_x, step_y = ((edge_point[0] - x) * speed // 8, (edge_point[1] - y) * speed // 8)
        points = [(x + i * step_x, y + i * step_y) for i in range(8)]

        return points

    def drawStarGaus(self, star, image):

        sigma = (star.fwhm / 2.355)
        sigma2 = sigma ** 2
        k = 1 / 2 / np.pi / sigma2
        lim = np.ceil(5 * sigma)

        upy = math.floor(max(0, star.y - lim))
        dwy = math.ceil(min(self.config.SizeY - 1, star.y + lim))

        upx = math.floor(max(0, star.x - lim))
        dwx = math.ceil(min(self.config.SizeX - 1, star.x + lim))

        for y in range(upy, dwy + 1):
            for x in range(upx, dwx):
                image[x, y] += star.brightness * self.bigaus(star.x - x + 0.5, star.y - y + 0.5, k, sigma2)

    def bigaus(self, x, y, k, sigma):
        return k * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)

    def addNoise(self, image):
        if self.config.Noise.enable:
            noise_image = np.abs(
                self.config.Noise.std * np.random.randn(self.config.SizeX, self.config.SizeY) + self.config.Noise.mean)
            image += noise_image

    def saveImgToFits(self, image, name):
        name = f'{name}.fits'
        fits.writeto(name, image.astype(np.float32), overwrite=True)


if __name__ == "__main__":
    config = configuration.loadConfig()

    gen = StarGenerator(config)

    gen.generateSeries()
