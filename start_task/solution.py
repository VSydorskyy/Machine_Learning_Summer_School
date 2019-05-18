import os
import argparse

import numpy as np

from PIL import Image

EPSELON = 2.
HIST_TRESH = 0.33
COR_TRESH = 0.3
HIST_NUM_BINS = 10


def read_images(image_dir: str) -> dict:
    """
    Read images from dir and aggregate them into dict
    :param image_dir: path to image dir
    :return: dict image_name:Pillow_object
    """
    image_names = os.listdir(image_dir)
    images = [Image.open(os.path.join(image_dir, image_name)) for image_name in image_names]
    return {k: i for k, i in zip(image_names, images)}


def check_duplicate(im_1: Image, im_2: Image) -> bool:
    """Check if images are complete duplicates (RGB arrays are equal)"""
    # If image size are different they are not duplicates
    if not np.array_equal(im_1.size, im_2.size):
        return False

    return np.array_equal(np.asarray(im_1), np.asarray(im_2))


def np_correlation(x_mass, y_mass):
    """
    Implementation of correlation distance from
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.correlation.html
    """
    centr_x = x_mass - x_mass.mean()
    centr_y = y_mass - y_mass.mean()

    return 1 - centr_x @ centr_y / (np.linalg.norm(centr_x) * np.linalg.norm(centr_y))


def mode_is_near_mean(distrib) -> bool:
    """
    Check if distribution mean is in bin of distribution mode
    If mode boarder bin is higher then HIST_TRESH it is accepted as mode bin
    """
    distrib_hist_y, distrib_hist_x = np.histogram(distrib, bins=HIST_NUM_BINS)
    distrib_hist_y = distrib_hist_y / distrib_hist_y.sum()

    mod_value_idx = distrib_hist_y.argmax()

    if mod_value_idx > 0 and distrib_hist_y[mod_value_idx - 1] > HIST_TRESH:
        return distrib_hist_x[mod_value_idx - 1] < distrib.mean() < distrib_hist_x[mod_value_idx + 1]

    elif mod_value_idx < HIST_NUM_BINS - 1 and distrib_hist_y[mod_value_idx + 1] > HIST_TRESH:
        return distrib_hist_x[mod_value_idx - 1] < distrib.mean() < distrib_hist_x[mod_value_idx + 2]

    else:
        return distrib_hist_x[mod_value_idx] < distrib.mean() < distrib_hist_x[mod_value_idx + 1]


def check_modify_and_similar(im_1: Image, im_2: Image, eps: float) -> bool:
    """Check if images are similar or modified"""
    if not np.array_equal(im_1.size, im_2.size):
        im_2 = im_2.resize(im_1.size)

    im_1 = np.asarray(im_1).astype(float)
    im_2 = np.asarray(im_2).astype(float)

    '''
    Correlation threshold is unreliable  
    if np_correlation(im_1.flatten(), im_2.flatten()) < COR_TRESH:
        return True
    '''

    im_differ = im_1 - im_2

    if mode_is_near_mean(im_differ[:, :, 0].flatten()) and \
            mode_is_near_mean(im_differ[:, :, 1].flatten()) and \
            mode_is_near_mean(im_differ[:, :, 2].flatten()):

        r_mean = im_differ[:, :, 0].flatten().mean()
        g_mean = im_differ[:, :, 1].flatten().mean()
        b_mean = im_differ[:, :, 2].flatten().mean()

        # Check if channels' means are in one epselon neighborhood
        return g_mean + eps > r_mean > g_mean - eps and \
               g_mean + eps > b_mean > g_mean - eps

    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find duplicated, modified, similar images')
    parser.add_argument('--path', type=str, help='path to file with images', default='start_task/data')
    args = parser.parse_args()

    image_dict = read_images(args.path)

    found_pairs = []

    for image_name_1 in image_dict.keys():

        duplicate = []
        similar_modify = []

        for image_name_2 in image_dict.keys():

            # Do not check images with the same name
            if image_name_1 == image_name_2:
                continue

            # Do not check images if we have them in 'positive checked' pairs
            if [image_name_2, image_name_1] in found_pairs:
                continue

            if check_duplicate(image_dict[image_name_1], image_dict[image_name_2]):
                duplicate.append(image_name_2)
                found_pairs.append([image_name_1, image_name_2])
                continue

            if check_modify_and_similar(image_dict[image_name_1], image_dict[image_name_2], eps=EPSELON):
                similar_modify.append(image_name_2)
                found_pairs.append([image_name_1, image_name_2])

        if len(duplicate) > 0:
            print('Duplicated images of {} image: {}'.format(image_name_1, (', ').join(duplicate)))

        if len(similar_modify) > 0:
            print('Similar or Modified images of {} image: {}'.format(image_name_1, (', ').join(similar_modify)))
