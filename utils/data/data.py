import os
from collections import defaultdict

import numpy as np
import skimage.util
import scipy.spatial.distance as spdist
from utils.image_utils import load_img, list_images
from sklearn.model_selection import train_test_split

from utils.params import (
    IMG_TARGET_ROWS, IMG_TARGET_COLS,
    DATA_PATH, IMG_ORIG_ROWS, IMG_ORIG_COLS
)


def _compute_img_hist(img):
    # Divide the image in blocks and compute per-block histogram
    blocks = skimage.util.view_as_blocks(img, block_shape=(20, 20))
    img_hists = [np.histogram(block, bins=np.linspace(0, 1, 10))[0] for block in blocks]
    return np.concatenate(img_hists)


def _are_inconsistent(mask1, mask2):
    has_mask1 = np.count_nonzero(mask1) > 0
    has_mask2 = np.count_nonzero(mask2) > 0
    return has_mask1 != has_mask2


def _filter_inconsistent(imgs, masks):
    hists = np.array(map(_compute_img_hist, imgs))
    dists = spdist.squareform(spdist.pdist(hists, metric='cosine'))

    # + eye because image will be similar to itself. We dont want to include those.
    close_pairs = dists + np.eye(dists.shape[0]) < 0.008
    close_ij = np.transpose(np.nonzero(close_pairs))

    # Find inconsistent masks among duplicates
    valids = np.ones(len(imgs), dtype=np.bool)
    for i, j in close_ij:
        if _are_inconsistent(masks[i], masks[j]):
            valids[i] = valids[j] = False

    return np.array(imgs)[valids], np.array(masks)[valids]


class DataManager(object):
    @staticmethod
    def read_train_images():
        train_data_path = os.path.join(DATA_PATH, 'train')
        images = list_images(train_data_path)
        total = int(len(images) / 2)

        patient_classes = np.ndarray(total, dtype=np.uint8)
        imgs = np.ndarray((total, IMG_ORIG_ROWS, IMG_ORIG_COLS), dtype=np.uint8)
        imgs_mask = np.ndarray((total, IMG_ORIG_ROWS, IMG_ORIG_COLS), dtype=np.uint8)

        print('Loading training images...')
        i = 0
        for image_path in images:
            if 'mask' in image_path:
                continue

            image_name = os.path.basename(image_path)
            name = image_name.split('.')[0]
            patient_classes[i] = int(name.split('_')[0])

            image_mask_name = name + '_mask.tif'
            imgs[i] = load_img(os.path.join(train_data_path, image_name), grayscale=True)
            imgs_mask[i] = load_img(os.path.join(train_data_path, image_mask_name), grayscale=True)

            if i % 100 == 0:
                print(f'Done: {i}/{total} images')
            i += 1
        return patient_classes, imgs, imgs_mask

    @staticmethod
    def create_train_data():
        patient_classes, imgs, imgs_mask = DataManager.read_train_images()

        print('Creating train dataset...')
        mask_labels = [1 if np.count_nonzero(mask) > 0 else 0 for mask in imgs_mask]
        DataManager.save_train_val_split(imgs, imgs_mask, "all", stratify=mask_labels)

    @staticmethod
    def create_cleaned_train_data():
        # Group by patient id.
        patient_classes, imgs, imgs_mask = DataManager.read_train_images()

        print("Cleaning bad training data...")
        pid_data_dict = defaultdict(list)
        for i, pid in enumerate(patient_classes):
            pid_data_dict[pid].append((imgs[i], imgs_mask[i]))

        imgs_cleaned = []
        imgs_masks_cleaned = []
        for pid in pid_data_dict:
            imgs, masks = zip(*pid_data_dict[pid])
            filtered_imgs, filtered_masks = _filter_inconsistent(imgs, masks)
            print("Discarded {} from patient {}".format(len(imgs) - len(filtered_imgs), pid))
            imgs_cleaned.extend(filtered_imgs)
            imgs_masks_cleaned.extend(filtered_masks)

        imgs = np.array(imgs_cleaned)
        imgs_mask = np.array(imgs_masks_cleaned)
        print("Creating cleaned train dataset: {} items".format(len(imgs)))
        mask_labels = [1 if np.count_nonzero(mask) > 0 else 0 for mask in imgs_mask]
        DataManager.save_train_val_split(imgs, imgs_mask, "cleaned", stratify=mask_labels)

    @staticmethod
    def create_test_data():
        train_data_path = os.path.join(DATA_PATH, 'test')
        images = os.listdir(train_data_path)
        total = len(images)

        imgs = np.ndarray((total, 1, IMG_ORIG_ROWS, IMG_ORIG_COLS), dtype=np.uint8)
        imgs_id = np.ndarray((total, ), dtype=np.int32)

        print('Creating test images...')
        i = 0
        for image_path in images:
            image_name = os.path.basename(image_path)
            img_id = int(image_name.split('.')[0])
            img = load_img(os.path.join(train_data_path, image_name), grayscale=True)
            img = np.array([img])

            imgs[i] = img
            imgs_id[i] = img_id

            if i % 100 == 0:
                print(f'Done: {i}/{total} images')
            i += 1

        # Build all data set
        print('Saving test samples...')
        imgs = imgs[np.argsort(imgs_id)]
        np.save(os.path.join(DATA_PATH, 'imgs_test.npy'), imgs)
        print('Saving to .npy files done.')


    @staticmethod
    def save_train_val_split(X, y, name_prefix, stratify=None, split_ratio=0.1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=stratify, test_size=split_ratio)
        np.save(os.path.join(DATA_PATH, f'{name_prefix}_X_train.npy'), X_train)
        np.save(os.path.join(DATA_PATH, f'{name_prefix}_X_val.npy'), X_val)
        np.save(os.path.join(DATA_PATH, f'{name_prefix}_y_train.npy'), y_train)
        np.save(os.path.join(DATA_PATH, f'{name_prefix}_y_val.npy'), y_val)
        print(f'Saving {name_prefix} .npy files done.')


    @staticmethod
    def load_train_val_data(name_prefix):
        X_train = np.load(os.path.join(DATA_PATH, f'{name_prefix}_X_train.npy'))
        X_val = np.load(os.path.join(DATA_PATH, f'{name_prefix}_X_val.npy'))
        y_train = np.load(os.path.join(DATA_PATH, f'{name_prefix}_y_train.npy'))
        y_val = np.load(os.path.join(DATA_PATH, f'{name_prefix}_y_val.npy'))
        return X_train, X_val, y_train, y_val


    @staticmethod
    def load_test_data():
        return np.load(os.path.join(DATA_PATH, 'imgs_test.npy'))
