from utils import open_file
import numpy as np
import cv2




CUSTOM_DATASETS_CONFIG = {
         'PCA_PaviaU_50': {
            'img': 'PCA_PaviaU_50.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: PCA_PaviaU_50_loader(folder)
            }
    }


def PCA_PaviaU_50_loader(folder):
        img = open_file(folder + 'PCA_PaviaU_50.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (49, 41, 12)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette

CUSTOM_DATASETS_CONFIG = {
         'PCA_PaviaU_20': {
            'img': 'PCA_PaviaU_20.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: PCA_PaviaU_20_loader(folder)
            }
    }


def PCA_PaviaU_20_loader(folder):
        img = open_file(folder + 'PCA_PaviaU_20.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (10, 11, 12)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'PCA_PaviaU_10': {
            'img': 'PCA_PaviaU_10.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: PCA_PaviaU_10_loader(folder)
            }
    }


def PCA_PaviaU_10_loader(folder):
        img = open_file(folder + 'PCA_PaviaU_10.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (7, 8, 9)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette



CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_85': {
            'img': 'unPCA_PaviaU_85.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_85_loader(folder)
            }
    }


def unPCA_PaviaU_85_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_85.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette

CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_90': {
            'img': 'unPCA_PaviaU_90.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_90_loader(folder)
            }
    }


def unPCA_PaviaU_90_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_90.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette

CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_95': {
            'img': 'unPCA_PaviaU_95.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_95_loader(folder)
            }
    }


def unPCA_PaviaU_95_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_95.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette



CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_80': {
            'img': 'unPCA_PaviaU_80.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_80_loader(folder)
            }
    }


def unPCA_PaviaU_80_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_80.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette



CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_75': {
            'img': 'unPCA_PaviaU_75.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_75_loader(folder)
            }
    }


def unPCA_PaviaU_75_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_75.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette



CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_70': {
            'img': 'unPCA_PaviaU_70.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_70_loader(folder)
            }
    }


def unPCA_PaviaU_70_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_70.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_60': {
            'img': 'unPCA_PaviaU_60.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_60_loader(folder)
            }
    }


def unPCA_PaviaU_60_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_60.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_65': {
            'img': 'unPCA_PaviaU_65.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_65_loader(folder)
            }
    }


def unPCA_PaviaU_65_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_65.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_55': {
            'img': 'unPCA_PaviaU_55.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_55_loader(folder)
            }
    }


def unPCA_PaviaU_55_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_55.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_50': {
            'img': 'unPCA_PaviaU_50.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_50_loader(folder)
            }
    }


def unPCA_PaviaU_50_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_50.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_45': {
            'img': 'unPCA_PaviaU_45.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_45_loader(folder)
            }
    }


def unPCA_PaviaU_45_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_45.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette



CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_40': {
            'img': 'unPCA_PaviaU_40.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_40_loader(folder)
            }
    }


def unPCA_PaviaU_40_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_40.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette



CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_35': {
            'img': 'unPCA_PaviaU_35.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_35_loader(folder)
            }
    }


def unPCA_PaviaU_35_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_35.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_30': {
            'img': 'unPCA_PaviaU_30.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_30_loader(folder)
            }
    }


def unPCA_PaviaU_30_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_30.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_25': {
            'img': 'unPCA_PaviaU_25.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_25_loader(folder)
            }
    }


def unPCA_PaviaU_25_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_25.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_20': {
            'img': 'unPCA_PaviaU_20.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_20_loader(folder)
            }
    }


def unPCA_PaviaU_20_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_20.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_15': {
            'img': 'unPCA_PaviaU_15.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_15_loader(folder)
            }
    }


def unPCA_PaviaU_15_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_15.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_10': {
            'img': 'unPCA_PaviaU_10.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_10_loader(folder)
            }
    }


def unPCA_PaviaU_10_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_10.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_5': {
            'img': 'unPCA_PaviaU_5.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_5_loader(folder)
            }
    }


def unPCA_PaviaU_5_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_5.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette




CUSTOM_DATASETS_CONFIG = {
         'PCA_PaviaU_10': {
            'img': 'PCA_PaviaU_10.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: PCA_PaviaU_10_loader(folder)
            }
    }

def PCA_PaviaU_10_loader(folder):
        img = open_file(folder + 'PCA_PaviaU_10.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'PCA_PaviaU_3': {
            'img': 'PCA_PaviaU_3.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: PCA_PaviaU_3_loader(folder)
            }
    }


def PCA_PaviaU_3_loader(folder):
        img = open_file(folder + 'PCA_PaviaU_3.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'unPCA_PaviaU_100': {
            'img': 'unPCA_PaviaU_100.mat',
            'gt': 'PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: unPCA_PaviaU_100_loader(folder)
            }
    }


def unPCA_PaviaU_100_loader(folder):
        img = open_file(folder + 'unPCA_PaviaU_100.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (0, 1, 2)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'Augmented_IndianPines': {
            'img': 'Augmented_IndianPines.mat',
            'gt': 'Augmented_IndianPines_gt.mat',
            'download': False,
            'loader': lambda folder: Augmented_IndianPines_loader(folder)
            }
    }


def Augmented_IndianPines_loader(folder):
        img = open_file(folder + 'Augmented_IndianPines.mat')[:,:,:-2]
        gt = open_file(folder + 'Augmented_IndianPines_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (43, 21, 11)

        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette




CUSTOM_DATASETS_CONFIG = {
         'Augmented_PaviaU': {
            'img': 'Augmented_PaviaU.mat',
            'gt': 'Augmented_PaviaU_gt.mat',
            'download': False,
            'loader': lambda folder: Augmented_PaviaU_loader(folder)
            }
    }

def Augmented_PaviaU_loader(folder):
        img = open_file(folder + 'Augmented_PaviaU.mat')[:,:,:-2]
        gt = open_file(folder + 'Augmented_PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (43, 21, 11)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette



CUSTOM_DATASETS_CONFIG = {
         'Augmented_Mississippi_Gulfport': {
            'img': 'MY_MG.mat',
            'gt': 'MY_MG_gt.mat',
            'download': False,
            'loader': lambda folder: Augmented_Mississippi_Gulfport_loader(folder)
            }
    }

def Augmented_Mississippi_Gulfport_loader(folder):
        img = open_file(folder + 'MY_MG.mat')[:,:,:-2]
        gt = open_file(folder + 'MY_MG_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (43, 21, 11)

        label_values = ['Undefined', 'Trees', 'Grass ground', 'Mixed ground', 'Dirt and sand',
                        'Road', 'Water', 'Buildings',
                        'Shadows of buildings', 'Sidewalk','Yellow curb','Cloth panels']
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette



CUSTOM_DATASETS_CONFIG = {
         'Mississippi_Gulfport': {
            'img': 'MG.mat',
            'gt': 'MG_gt.mat',
            'download': False,
            'loader': lambda folder: Mississippi_Gulfport_loader(folder)
            }
    }

def Mississippi_Gulfport_loader(folder):
        img = open_file(folder + 'MG.mat')[:,:,:-2]
        gt = open_file(folder + 'MG_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (43, 21, 11)

        label_values = ['Undefined', 'Trees', 'Grass ground', 'Mixed ground', 'Dirt and sand',
                        'Road', 'Water', 'Buildings',
                        'Shadows of buildings', 'Sidewalk','Yellow curb','Cloth panels']
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'Simulate_database': {
            'img': 'Simu_data.mat',
            'gt': 'Simu_label.mat',
            'download': False,
            'loader': lambda folder: Simulate_database_loader(folder)
            }
    }

def Simulate_database_loader(folder):
        img = open_file(folder + 'Simu_data.mat')[:,:,:-2]
        gt = open_file(folder + 'Simu_label.mat')
        gt = gt.astype('uint8')

        rgb_bands = (43, 21, 11)

        label_values = ['Undefined', 'C1','C2','C3','C4','C5']
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
    'new_PU': {
        'img': 'new_PU.mat',
        'gt': 'PaviaU_gt.mat',
        'download': False,
        'loader': lambda folder: new_PU_loader(folder)
            }
    }


def new_PU_loader(folder):
        img = open_file(folder + 'new_PU.mat')[:,:,:-2]
        gt = open_file(folder + 'PaviaU_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (43, 21, 11)

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette

CUSTOM_DATASETS_CONFIG = {
         'new_IP': {
            'img': 'new_IP.mat',
            'gt': 'Indian_pines_gt.mat',
            'download': False,
            'loader': lambda folder: new_IP_loader(folder)
            }
    }


def new_IP_loader(folder):
        img = open_file(folder + 'new_IP.mat')[:,:,:-2]
        gt = open_file(folder + 'Indian_pines_gt.mat')
        gt = gt.astype('uint8')

        rgb_bands = (43, 21, 11)

        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'new_SA': {
            'img': 'new_SA.mat',
            'gt': 'Salinas_gt.mat',
            'download': False,
            'loader': lambda folder: new_SA_loader(folder)
            }
         }


def new_SA_loader(folder):
    img = open_file(folder + 'new_SA.mat')[:, :, :-2]
    gt = open_file(folder + 'Salinas_gt.mat')
    gt = gt.astype('uint8')

    rgb_bands = (75, 5, 169)

    label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                    'Fallow_smooth',
                    'Stubble', 'Celery', 'Grapes_untrained',
                    'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk',
                    'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']

    ignored_labels = [0]
    return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'Salinas': {
            'img': 'Salinas_corrected.mat',
            'gt': 'Salinas_gt.mat',
            'download': False,
            'loader': lambda folder: Salinas_loader(folder)
            }
    }

def Salinas_loader(folder):
        img = open_file(folder + 'Salinas_corrected.mat')[:,:,:-2]
        gt = open_file(folder + 'Salinas_gt.mat')['salinas_gt']
        gt = gt.astype('uint8')

        rgb_bands = (75, 5, 169)

        label_values = ['Undefined','Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained',
                        'Soil_vinyard_develop', 'Corn_senesced_green_weeds','Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis']

        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'DFC2018_HSI': {
            'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
            'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif',
            'download': False,
            'loader': lambda folder: DFC2018_HSI_loader(folder)
            }
    }


def DFC2018_HSI_loader(folder):
        img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')[:,:,:-2]
        gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
        gt = gt.astype('uint8')
        # The original data img size(601, 2384, 50) and gt size(1202, 4768)
        # So you first need to downsample the img data or upsample the gt data
        gt = cv2.resize(gt, dsize=(img.shape[0],img.shape[1]), interpolation=cv2.INTER_NEAREST)
        # img  = cv2.resize(img, dsize=(gt.shape[0],gt.shape[1]), interpolation=cv2.INTER_CUBIC)

        rgb_bands = (47, 31, 15)

        label_values = ["Unclassified",
                        "Healthy grass",
                        "Stressed grass",
                        "Artificial turf",
                        "Evergreen trees",
                        "Deciduous trees",
                        "Bare earth",
                        "Water",
                        "Residential buildings",
                        "Non-residential buildings",
                        "Roads",
                        "Sidewalks",
                        "Crosswalks",
                        "Major thoroughfares",
                        "Highways",
                        "Railways",
                        "Paved parking lots",
                        "Unpaved parking lots",
                        "Cars",
                        "Trains",
                        "Stadium seats"]
        ignored_labels = [0]
        # palette = None
        return img, gt, rgb_bands, ignored_labels, label_values, palette

CUSTOM_DATASETS_CONFIG = {
         'Houston': {
            'img': 'Houston.mat',
            'gt': 'Houston_gt.mat',
            'download': False,
            'loader': lambda folder: Houston_loader(folder)
            }
    }

def Houston_loader(folder):
        img = open_file(folder + 'Houston.mat')[:,:,:-2]
        gt = open_file(folder + 'Houston_gt.mat')['Houston_gt']
        gt = gt.astype('uint8')

        rgb_bands = (59, 40, 23)

        label_values = ['Undefined','Healthy grass','Stressed grass','Synthetic grass','Trees',
         'Soil','Water','Residential','Commercial','Road','Highway',
         'Railway','Parking Lot1','Parking Lot2','Tennis court','Running track']

        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette


