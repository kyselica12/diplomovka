from dataclasses import dataclass
import random
import yaml


@dataclass
class Value:
    min: int
    max: int
    random: str

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def value(self):
        if self.random == 'uniform':
            return random.randrange(self.min, self.max+1)
        if self.random == 'normal':
            return int(random.gauss(mu=(abs(self.min + self.max) / 2), sigma=(self.max - self.min)))

@dataclass
class Noise:
    mean: int
    std: int
    enable: bool

    def __init__(self, **entries):
        self.__dict__.update(entries)


@dataclass
class Stars:
    count: Value
    brightness: Value
    fwhm: Value

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = Value(**self.count)
        self.brightness = Value(**self.brightness)
        self.fwhm = Value(**self.fwhm)


@dataclass
class Objects:
    count: Value
    brightness: Value
    fwhm: Value
    speed: Value

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.count = Value(**self.count)
        self.brightness = Value(**self.brightness)
        self.fwhm = Value(**self.fwhm)
        self.speed = Value(**self.speed)


@dataclass
class Configuration:
    numberOfSeries : int
    SizeX: int
    SizeY: int
    Stars: Stars
    Objects: Objects
    dataFile: str
    plot: bool
    save: bool
    Noise: Noise

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.Stars = Stars(**self.Stars)
        self.Objects = Objects(**self.Objects)
        self.Noise = Noise(**self.Noise)


def loadConfig():
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    return Configuration(**cfg)


if __name__ == "__main__":
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    c = Configuration(**cfg)

    # print(c)
