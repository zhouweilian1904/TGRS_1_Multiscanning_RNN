# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import itertools
import spectral
import visdom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import io, misc
import os
import re
import torch
import imageio
import math
# sns.set_style('darkgrid')
import cv2
from PIL import Image
import io as io2

def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def convert_to_color_(arr_2d, palette=None):
    # print('arr_2d:', type(arr_2d), arr_2d.shape)
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)), np.transpose(gt, (2, 0, 1))], nrow=2,  opts={'caption': caption})


def display_dataset(img, gt, bands, labels, palette, vis):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    print('Image shape:', img.shape) #(h w c)
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')
    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})

    #inter-band cross correlation
    height, width, bands = img.shape
    # Reshape the image to be (bands, height*width)
    reshaped_image = np.reshape(img, (height * width, bands)).T
    # Calculate the correlation coefficient matrix
    corr_matrix = np.corrcoef(reshaped_image) #it's a measure of the linear relationship between two datasets.
    corr_matrix_torch = torch.from_numpy(corr_matrix)
    #create a heatmap
    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix_torch, cmap='jet', cbar_kws={'label': 'Pearson correlation coefficient'}, linewidths=0, linecolor='black')
    plt.title('Inter-band cross correlation', fontsize=15)  # you can set title size here
    plt.xlabel('Band Number', fontsize=15)
    plt.ylabel('Band Number', fontsize=15)
    plt.gca().invert_yaxis()
    vis.matplot(plt)


def explore_spectrums(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c], fontsize=15)
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.3)
        mean_spectrum = np.mean(class_spectrums, axis=0)

        std_spectrum = np.std(class_spectrums, axis=0)
        # print('type(std_spect)',type(std_spectrum))
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        # plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
        #                  higher_spectrum, color="#3F5D7D")
        # plt.plot(std_spectrum, alpha=1, color="blue", lw=2.5)
        # plt.plot(higher_spectrum, alpha=1, color="red", lw=2.5)
        # plt.plot(lower_spectrum, alpha=1, color="black", lw=2.5)
        plt.plot(mean_spectrum, alpha=1, color="red", lw=4)
        plt.xticks(fontsize=15)  # Increase x-axis tick label font size to 12
        plt.ylabel('Spectral Value', fontsize=15)
        plt.yticks(fontsize=15)  # Increase y-axis tick label font size to 12
        plt.xlabel('Band Number',fontsize=15)
        # Adding the bounding box
        ax = plt.gca()
        box = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=None, edgecolor='black', linewidth=5)
        ax.add_patch(box)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums

def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
    """
    plt.figure(figsize=(10, 7))
    for k, v in spectrums.items():
        sns.lineplot(x=range(len(v)), y=v, label=k)
    plt.title(title,fontsize=20 )
    plt.xticks(fontsize=20)  # Increase x-axis tick label font size to 12
    plt.ylabel('Spectral Value', fontsize=20)
    plt.yticks(fontsize=20)  # Increase y-axis tick label font size to 12
    plt.xlabel('Band Number', fontsize=20)
    # Adding the bounding box
    ax = plt.gca()
    box = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=None, edgecolor='black', linewidth=5)
    ax.add_patch(box)
    plt.legend(fontsize='medium')
    vis.matplot(plt)

def calculate_psnr(img1, img2):
    #A higher PSNR value indicates a smaller amount of distortion or noise in the reconstructed or compressed signal,
    # while a lower PSNR value indicates higher distortion or noise.
    mse = np.mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = np.amax(img1)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    accuracy = accuracy

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results


def show_results(results, vis, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]

    vis.heatmap(cm, opts={'title': "Confusion matrix", 
                          'marginbottom': 100,
                          'marginleft': 100,
                          'width': 800,
                          'height': 800,
                          'rownames': label_values, 'columnnames': label_values,
                          'colormap': 'Jet'})
    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.05f} +- {:.05f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.05f}%\n".format(accuracy)
    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.05f} +- {:.05f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.05f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.05f} +- {:.05f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.05f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print(text)


def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """

    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes

    train_gt = np.zeros_like(gt)
    print('train_gt',train_gt.shape)
    test_gt = np.zeros_like(gt)

    if train_size > 1:
       train_size = int(train_size)
    
    if mode == 'random':
       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       # train_gt[train_indices] = gt[train_indices]
       # test_gt[test_indices] = gt[test_indices]
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       # train_gt[train_indices] = gt[train_indices]
       # test_gt[test_indices] = gt[test_indices]
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients 
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def spiral_order(matrix):
    res = []
    while matrix:
        res += matrix.pop(0)
        matrix = list(zip(*matrix))[::-1]
    return res

def perimeter_scan_batch(batch):
    batch_size, _, height, width = batch.shape
    batch_sequence = []

    for b in range(batch_size):
        img = batch[b].detach().numpy()

        sequence = np.zeros((img.shape[0], height * width))

        for c in range(img.shape[0]):
            sequence[c, :] = spiral_order(img[c, :, :].tolist())

        batch_sequence.append(sequence)

    batch_sequence = torch.stack([torch.from_numpy(seq) for seq in batch_sequence])

    return batch_sequence


def diagonal_flatten_image(image):
        device = image.device  # Get the device of the input tensor
        batch_size, channels, width, height = image.size()

        # Move the image tensor to the CPU if it's on GPU
        if device.type == 'cuda':
            image = image.cpu()

        # Reshape the image by swapping width and height
        image = image.transpose(2, 3).reshape(batch_size, channels, -1)  # (batch, channels, width * height)

        # Calculate the length of the sequence
        length = width * height

        # Generate the indices to flatten the image diagonally
        indices = torch.arange(length)
        indices = indices.reshape(1, 1, length)  # (1, 1, length)

        # Expand the indices to match the batch and channel dimensions
        indices = indices.expand(batch_size, channels, length)  # (batch, channels, length)

        # Move the indices tensor back to the original device
        indices = indices.to(device)

        # Gather the values from the image using the indices
        flattened_image = torch.gather(image, dim=2, index=indices)

        return flattened_image