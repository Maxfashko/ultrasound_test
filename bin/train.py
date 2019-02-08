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
run_id = str(datetime.now())
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

    ## concat all data for kfold
    #X_train = np.concatenate((X_train, X_val), axis=0)
    #y_train = np.concatenate((y_train, y_val), axis=0)
#
    #kf = KFold(n_splits=N_FOLDS)
#
    #for train_index, test_index in kf.split(X):
    #    X_train, X_test = X_train[train_index], X_train[test_index]
    #    y_train, y_test = y_train[train_index], y_train[test_index]

    #N_FOLDS
    print('Creating and compiling model...')
    model = build_model()
    #if resume:
    #    model.load_weights('./results/net.hdf5')


    print('Training on model')
    #model.summary()
    batch_size = 64
    epochs = 200

    tb = TensorBoard(
        log_dir                = f'./logs/{run_id}',
        histogram_freq         = 0,
        write_graph            = True,
        write_grads            = False,
        write_images           = False,
        embeddings_freq        = 0,
        embeddings_layer_names = None,
        embeddings_metadata    = None
    )

    clr_triangular = CyclicLR(mode='triangular', base_lr=1e-7, max_lr=1e-3, step_size=epochs)
    early_s = EarlyStopping(monitor='val_main_output_dice', patience=55, verbose=1)

    # проблемы с сохранением лучших весов по метрике val_dice в кастомном колбэке
    model_checkpoint = MyModelCheck('./results/net.hdf5',  mode='max', monitor='val_loss', save_best_only=True, save_weights_only=False)

    # костыль для TensorBoard, без него не показывается lr
    reduce_lr = ReduceLROnPlateau(monitor='val_main_output_dice', factor=0.2, patience=10, min_lr=1e-6)

    train_generator = DataGenerator(
        list_IDs=X_train.shape[0],
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        transform=lambda x, y: transform(x, y, augment=True)
    )

    # TTA for validation = result bellow
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
        callbacks=[model_checkpoint, tb, clr_triangular, reduce_lr, early_s]
    )


if __name__ == '__main__':
    train(resume=False)