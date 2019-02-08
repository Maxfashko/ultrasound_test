import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, X, y, transform, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.transform = transform
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.list_IDs / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        inputs, outputs = self.__data_generation(list_IDs_temp)
        return inputs, outputs


    def on_epoch_end(self):
        self.indexes = np.arange(self.list_IDs)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        batch_img = []
        batch_has_mask = []
        batch_mask = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img, lbl = self.transform(self.X[ID], self.y[ID])
            batch_img.append(img)
            batch_has_mask.append(1 if np.count_nonzero(self.y[ID]) > 0 else 0)
            batch_mask.append(lbl)

        inputs = np.array(batch_img)
        outputs = {
            'aux_output': np.array(batch_has_mask),
            'main_output': np.array(batch_mask)
        }
        return inputs, outputs
