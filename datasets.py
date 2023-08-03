# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
import cv2
from sklearn.decomposition import PCA
from tqdm import tqdm
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils import open_file

DATASETS_CONFIG = {
        'DFC2018_HSI': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                    'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
            'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif'
            },
        'PaviaC': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat', 
                     'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
            'img': 'Pavia.mat',
            'gt': 'Pavia_gt.mat'
            },
        'PaviaU': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'PaviaU.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'new_PU': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'new_PU.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'PCA_PaviaU_3': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'PCA_PaviaU_3.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'PCA_PaviaU_10': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'PCA_PaviaU_10.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'Augmented_IndianPines': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                     'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
            'img': 'Augmented_IndianPines.mat',
            'gt': 'Augmented_IndianPines_gt.mat'
            },
        'Augmented_PaviaU': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'Augmented_PaviaU.mat',
            'gt': 'Augmented_PaviaU_gt.mat'
            },
         'Augmented_Mississippi_Gulfport': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'MY_MG.mat',
            'gt': 'MY_MG_gt.mat'
            },
         'Mississippi_Gulfport': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'MG.mat',
            'gt': 'MG_gt.mat'
            },
         'Simulate_database': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'Simu_data.mat',
            'gt': 'Simu_label.mat'
            },
         'Salinas': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'Salinas_corrected.mat',
            'gt': 'Salinas_gt.mat'
            },
         'new_SA': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'new_SA.mat',
            'gt': 'Salinas_gt.mat'
          },
         'Houston': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'Houston.mat',
            'gt': 'Houston_gt.mat'
            },
        'unPCA_PaviaU_100':{
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'unPCA_PaviaU_100.mat',
            'gt': 'paviaU_gt.mat'
        },
        'KSC': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                     'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
            'img': 'KSC.mat',
            'gt': 'KSC_gt.mat'
            },
        'IndianPines': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                     'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
            'img': 'Indian_pines_corrected.mat',
            'gt': 'Indian_pines_gt.mat'
            },
        'new_IP': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
            'img': 'new_IP.mat',
            'gt': 'Indian_pines_gt.mat'
            },
        'Botswana': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                     'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
            'img': 'Botswana.mat',
            'gt': 'Botswana_gt.mat',
            }
    }

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', True):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                     reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Pavia_gt.mat')['pavia_gt']

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]

        ignored_labels = [0]

    elif dataset_name == 'PCA_PaviaU_3':
        # Load the image
        img = open_file(folder + 'PCA_PaviaU_3.mat')['PCA_PaviaU_3']

        rgb_bands = (0,1,2)

        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]


    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')['paviaU']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]
        # palette = {0: (0, 0, 0),
        #            1: (255, 0, 0),
        #             2: (0, 255, 0),
        #             3: (0, 0, 255),
        #             4: (255, 255, 0),
        #             5: (255, 0, 255),
        #             6: (0, 255, 255),
        #             7: (255, 165, 0),
        #             8: (128, 0, 128),
        #             9: (0, 128, 0),}


    elif dataset_name == 'IndianPines':
        # Load the image
        img = open_file(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']

        # rgb_bands = (35, 24, 5)  # AVIRIS sensor
        rgb_bands = (27, 17, 7)  # AVIRIS sensor
        # rgb_bands = (29, 19, 9)  # AVIRIS sensor

        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']

        label_values = ["0.Undefined", "1.Alfalfa", "2.Corn-notill", "3.Corn-mintill",
                        "4.Corn", "5.Grass-pasture", "6.Grass-trees",
                        "7.Grass-pasture-mowed", "8.Hay-windrowed", "9.Oats",
                        "10.Soybean-notill", "11.Soybean-mintill", "12.Soybean-clean",
                        "13.Wheat", "14.Woods", "15.Buildings-Grass-Trees-Drives",
                        "16.Stone-Steel-Towers"]

        ignored_labels = [0]
        palette = {0: (0, 0, 0),
                   1: (255, 0, 0),
                    2: (0, 255, 0),
                    3: (0, 0, 255),
                    4: (255, 255, 0),
                    5: (255, 0, 255),
                    6: (0, 255, 255),
                    7: (255, 165, 0),
                    8: (128, 0, 128),
                    9: (0, 128, 0),
                    10: (128, 0, 0),
                    11: (128, 128, 0),
                    12: (0, 128, 128),
                    13: (0, 0, 128),
                    14: (255, 255, 255),
                    15: (128, 128, 128),
                    16: (165, 165, 165)
                   }

    elif dataset_name == 'new_IP':
        # Load the image
        img = open_file(folder + 'new_IP.mat')
        img = img['new_IP']

        rgb_bands = (35, 24, 5)  # AVIRIS sensor

        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']

        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Augmented_IndianPines':
        # Load the image
        img = open_file(folder + 'Augmented_IndianPines.mat')['MY_Indian_pines']
        # img = img['Augmented_IndianPines']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'Augmented_IndianPines_gt.mat')['MY_Label']

        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Augmented_PaviaU':
        # Load the image
        img = open_file(folder + 'Augmented_PaviaU.mat')['MY_paviaU']
        # img = img['Augmented_IndianPines']

        rgb_bands = (55, 41, 12)  # AVIRIS sensor

        gt = open_file(folder + 'Augmented_PaviaU_gt.mat')['MY_paviaU_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'Augmented_Mississippi_Gulfport':
        # Load the image
        img = open_file(folder + 'MY_MG.mat')['MY_MG']
        # img = img['Augmented_IndianPines']

        rgb_bands = (55, 41, 12)  # AVIRIS sensor

        gt = open_file(folder + 'MY_MG_gt.mat')['MY_MG_gt']

        label_values = ['Undefined', 'Trees', 'Grass ground', 'Mixed ground', 'Dirt and sand',
                        'Road', 'Water', 'Buildings',
                        'Shadows of buildings', 'Sidewalk','Yellow curb','Cloth panels']

        ignored_labels = [0]

    elif dataset_name == 'Mississippi_Gulfport':
        # Load the image
        img = open_file(folder + 'MG.mat')['Data']
        # img = img['Augmented_IndianPines']

        rgb_bands = (55, 41, 12)  # AVIRIS sensor

        gt = open_file(folder + 'MG_gt.mat')['Label']

        label_values = ['Undefined', 'Trees', 'Grass ground', 'Mixed ground', 'Dirt and sand',
                        'Road', 'Water', 'Buildings',
                        'Shadows of buildings', 'Sidewalk','Yellow curb','Cloth panels']

        ignored_labels = [0]

    elif dataset_name == 'Simulate_database':
        # Load the image
        img = open_file(folder + 'Simu_data.mat')['Simu_data']
        # img = img['Augmented_IndianPines']

        rgb_bands = (55, 41, 12)  # AVIRIS sensor

        gt = open_file(folder + 'Simu_label.mat')['Simu_label']

        label_values = ['Undefined', 'C1','C2','C3','C4','C5']

        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = open_file(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]


    elif dataset_name == 'Salinas':

        img = open_file(folder + 'Salinas_corrected.mat')['salinas_corrected']

        rgb_bands = (43,21, 11)

        gt = open_file(folder + 'Salinas_gt.mat')['salinas_gt']

        label_values = ['Undefined','Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained',
                        'Soil_vinyard_develop', 'Corn_senesced_green_weeds','Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis']

        ignored_labels = [0]

    elif dataset_name == 'new_SA':

        img = open_file(folder + 'new_SA.mat')['new_SA']

        rgb_bands = (35,24, 5)

        gt = open_file(folder + 'Salinas_gt.mat')['salinas_gt']

        label_values = ['Undefined','Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow', 'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained',
                        'Soil_vinyard_develop', 'Corn_senesced_green_weeds','Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis']

        ignored_labels = [0]

    elif dataset_name == 'unPCA_PaviaU_100':

        img = open_file(folder + 'unPCA_PaviaU_100.mat')['unPCA_PaviaU_100']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ['0.Undefined', '1.Asphalt', '2.Meadows', '3.Gravel', '4.Trees',
                        '5.Painted metal sheets', '6.Bare Soil', '7.Bitumen',
                        '8.Self-Blocking Bricks', '9.Shadows']

        ignored_labels = [0]

    elif dataset_name == 'new_PU':
        # Load the image
        img = open_file(folder + 'new_PU.mat')['new_PU']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'Houston':
        # Load the image
        img = open_file(folder + 'Houston.mat')['Houston']

        # rgb_bands = (59, 40, 23)
        rgb_bands = (60,40,20)
        gt = open_file(folder + 'Houston_gt.mat')['Houston_gt']

        label_values = ['0.Undefined','1.Healthy grass','2.Stressed grass','3.Synthetic grass','4.Trees',
         '5.Soil','6.Water','7.Residential','8.Commercial','9.Road','10.Highway',
         '11.Railway','12.Parking Lot1','13.Parking Lot2','14.Tennis court','15.Running track']

        ignored_labels = [0]
        # palette = {0: (0, 0, 0),
        #            1: (255, 0, 0),
        #             2: (0, 255, 0),
        #             3: (0, 0, 255),
        #             4: (255, 255, 0),
        #             5: (255, 0, 255),
        #             6: (0, 255, 255),
        #             7: (255, 165, 0),
        #             8: (128, 0, 128),
        #             9: (0, 128, 0),
        #             10: (128, 0, 0),
        #             11: (128, 128, 0),
        #             12: (0, 128, 128),
        #             13: (0, 0, 128),
        #             14: (255, 255, 255)}

    elif dataset_name == 'DFC2018_HSI':
        # Load the image
        # img = open_file(folder + 'Houston.mat')['Houston']

        img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')
        print('img:',img.shape)
        gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
        print('gt',gt.shape)
        gt = cv2.resize(gt, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        print('gt', gt.shape)

        rgb_bands = (47, 31, 15)


        # gt = open_file(folder + 'Houston_gt.mat')['Houston_gt']

        label_values = ["0.Unclassified",
                        "1.Healthy grass",
                        "2.Stressed grass",
                        "3.Artificial turf",
                        "4.Evergreen trees",
                        "5.Deciduous trees",
                        "6.Bare earth",
                        "7.Water",
                        "8.Residential buildings",
                        "9.Non-residential buildings",
                        "10.Roads",
                        "11.Sidewalks",
                        "12.Crosswalks",
                        "12.Major thoroughfares",
                        "13.Highways",
                        "14.Railways",
                        "15.Paved parking lots",
                        "16.Unpaved parking lots",
                        "17.Cars",
                        "18.Trains",
                        "19.Stadium seats"]

        ignored_labels = [0]

    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)

    #PCA need?
    # img = applyPCA(img, 30)

    # # Convert the NumPy array to a PyTorch tensor
    # input_tensor = torch.from_numpy(np.transpose(img.astype(np.float32), (2, 0, 1))).unsqueeze(0).float()
    # # Adding Gaussian noise to the input image tensor
    # noisy_tensor = add_gaussian_noise(input_tensor, mean=0.0, std=0.1)
    # # Convert the noisy tensor back to a NumPy array
    # img = np.transpose(noisy_tensor.squeeze(0).numpy(), (1, 2, 0))

    # img = add_random_mask(img, 0.5)


    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    print('img shape', img.shape)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  ##之前是有的，记得点亮啊, 我只是在做tsne的时候灭掉了

    return img, gt, label_values, ignored_labels, rgb_bands, palette

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def add_gaussian_noise(image, mean=0.0, std=1.0):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return noisy_image

def add_random_mask(image, mask_ratio):
    mask = torch.rand(image.size()) < mask_ratio
    masked_image = image.clone()
    masked_image[mask] = 0  # Set masked values to 0 or any other desired value
    return masked_image

class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        #unsupervised:
        elif supervision == 'unsupervised':
            mask = np.ones_like(data)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
                data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label
