from datetime import datetime

import cv2
import numpy as np
from numpy.random import seed
from keras.callbacks import ReduceLROnPlateau, TensorBoard

from utils.params import N_FOLDS
from utils.clb.clr_callback import *
from utils.data.data import DataManager
from utils.augmenter import Augmentation
from sklearn.model_selection import KFold
from utils.model.model import build_model
from utils.data.generator import DataGenerator
from utils.clb.save_callback import MyModelCheck
from utils.image_utils import preprocess, norm, calc_mean_std


seed(1)
augmenter = Augmentation()


def transform(img, mask, augment=True):
    if augment:
        img, mask = augmenter.batch_augmentation(img, mask)
    img = preprocess(img)
    mask = preprocess(mask).astype('float32') / 255.
    return np.array([img]), np.array([mask])


def train(resume=False):
    print('Loading data...')
    X_train, X_val, y_train, y_val = DataManager.load_train_val_data("cleaned")
    n_fold = 0

    # concat all data for kfold
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    kf = KFold(n_splits=N_FOLDS)

    for train_index, test_index in kf.split(X):

        run_id = str(datetime.now())
        n_fold += 1

        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        print('Creating and compiling model...')
        model = build_model()

        weights_name = f'net_fold_{n_fold}.hdf5'

        print('Training on model')
        #model.summary()
        batch_size = 64
        epochs = 100

        tb = TensorBoard(
            log_dir                = f'./logs/{run_id}',
            histogram_freq         = 0,
            write_graph            = False,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )

        clr_triangular = CyclicLR(mode='triangular', base_lr=1e-7, max_lr=1e-3, step_size=epochs)
        #early_s = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

        # проблемы с сохранением лучших весов по метрике val_dice в кастомном колбэке
        model_checkpoint = MyModelCheck(f'./results/{weights_name}',  mode='max', monitor='val_loss', save_best_only=True, save_weights_only=False)

        # костыль для TensorBoard, без него не показывается lr
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, min_lr=1e-6)

        train_generator = DataGenerator(
            list_IDs=X_train.shape[0],
            X=X_train,
            y=y_train,
            batch_size=batch_size,
            transform=lambda x, y: transform(x, y, augment=True)
        )

        val_generator = DataGenerator(
            list_IDs=X_val.shape[0],
            X=X_val,
            y=y_val,
            batch_size=batch_size,
            transform=lambda x, y: transform(x, y, augment=True)
        )

        model.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            epochs=epochs,
            verbose=2,
            callbacks=[model_checkpoint, tb, clr_triangular, reduce_lr]
        )


if __name__ == '__main__':
    train(resume=False)
