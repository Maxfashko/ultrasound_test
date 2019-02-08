from itertools import chain
from datetime import datetime

import cv2
import numpy as np

from utils.params import (
    IMG_TARGET_ROWS,
    IMG_TARGET_COLS,
    IMG_ORIG_ROWS,
    IMG_ORIG_COLS,
    N_FOLDS)
from utils.data.data import DataManager
from utils.model.model import build_model
from utils.image_utils import preprocess, norm, calc_mean_std


def post_process_mask(prob_mask):
    prob_mask = prob_mask.astype('float32')
    prob_mask = cv2.resize(prob_mask, (IMG_ORIG_COLS, IMG_ORIG_ROWS),
                           interpolation=cv2.INTER_CUBIC)

    prob_mask = cv2.GaussianBlur(prob_mask, (1, 1), 0)

    median = cv2.medianBlur(prob_mask, 5)

    prob_mask = cv2.threshold(prob_mask, 0.4, 1, cv2.THRESH_BINARY)[1]
    return prob_mask


def run_length_enc(label):
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def generate_submission():
    X_train, _, _, _ = DataManager.load_train_val_data("cleaned")

    print('Loading and processing test images')
    imgs_test = DataManager.load_test_data()
    total = imgs_test.shape[0]
    imgs = np.ndarray((total, 1, IMG_TARGET_ROWS, IMG_TARGET_COLS), dtype=np.uint8)

    for idx, img in enumerate(imgs_test):
        img = img.reshape(IMG_ORIG_ROWS, IMG_ORIG_COLS)
        imgs[idx] = preprocess(img)

    model = build_model()

    masks_acc = np.ndarray((total, 1, IMG_TARGET_ROWS, IMG_TARGET_COLS), dtype=np.float32)
    has_masks_acc = np.ndarray((total, 1), dtype=np.float32)

    # predict using kfold
    for i in range(1, N_FOLDS+1):
        weights_name = f'net_fold_{i}.hdf5'

        print(f'Loading network for {weights_name}')

        model.load_weights(f'./results/{weights_name}')

        print('Generating predictions')
        masks, has_masks = model.predict(imgs, verbose=1)
        masks_acc += masks
        has_masks_acc += has_masks


    masks = masks_acc / N_FOLDS
    has_masks = has_masks_acc / 2

    ids = []
    rles = []
    for i in range(total):
        # Zero out masks when there is no-nerve pred.
        #if has_masks[i, 0] < 0.2:
        #    masks[i, 0] *= 0.

        # postprocess https://github.com/EdwardTyantov/ultrasound-nerve-segmentation
        new_prob = (has_masks[i, 0] + min(1, np.sum(masks[i, 0])/1000.0 )* 4 / 3)/2
        if np.sum(masks[i, 0]) > 0 and new_prob < 0.4:
            masks[i, 0] = np.zeros((IMG_TARGET_ROWS, IMG_TARGET_COLS))


        mask = post_process_mask(masks[i, 0])
        rle = run_length_enc(mask)
        rles.append(rle)
        ids.append(i + 1)

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = 'results/submission_{}.csv'.format(str(datetime.now()))

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


if __name__ == '__main__':
    generate_submission()
