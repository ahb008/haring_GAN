
import glob
from json import encoder
import imageio
import os

import numpy as np

from tensorflow.keras.layers import (Dense, Conv2D, Conv2DTranspose, Reshape,
                                     LeakyReLU, Activation, Lambda, BatchNormalization, GlobalAveragePooling2D, UpSampling2D, Flatten, MaxPooling2D, Input, Dropout, ZeroPadding2D)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, LambdaCallback

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



class Autoencoder:
    def __init__(self) -> None:
        self.image_width = 120
        self.image_height = 120
        self.image_channels = 3
        self.img_shape = (self.image_width, self.image_height, self.image_channels)
        self.latent_dim = 10

        self.encoder_model = self.build_encoder()
        self.decoder_model = self.build_decoder()

        input_image = Input(shape=self.img_shape)
        latent_space = self.encoder_model(input_image)
        generated_image = self.decoder_model(latent_space)

        self.full_model = Model(input_image, generated_image)
        self.full_model.compile(loss='mse',
                                optimizer=Adam(),
                                metrics=['mse', 'mae'])

    def build_encoder(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2,
                  input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(GlobalAveragePooling2D())
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.latent_dim))

        model.summary()

        img = Input(shape=self.img_shape)
        latent_space = model(img)

        return Model(img, latent_space)

    def build_decoder(self):
        model = Sequential()

        model.add(Dense(1024, input_shape=(self.latent_dim,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(15*15*8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((15, 15, 8)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(self.image_channels, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.image_channels, kernel_size=1, padding="same"))
        model.add(Activation('sigmoid'))

        model.summary()

        latent_space = Input(shape=(self.latent_dim,))
        img = model(latent_space)

        return Model(latent_space, img)

    def train(self, X_train, X_test, batch_size=32, epochs=10):
        self.full_model.fit(
            X_train, X_train,
            validation_data=(X_test, X_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: self.training_plot(epoch, X_test))]
            )

    def training_plot(self, epoch, X):
        n_samples = 5
        latent_space = self.encoder_model(X[:n_samples])
        gen_imgs = self.decoder_model.predict(latent_space)

        fig, axs = plt.subplots(3, n_samples)
        for i in range(n_samples):
            axs[0, i].imshow(X[i])
            axs[0, i].axis('off')
            axs[1, i].imshow(gen_imgs[i])
            axs[1, i].axis('off')
            axs[2, i].hist(latent_space.numpy().flatten())
        os.makedirs("./train_images", exist_ok=True)
        fig.savefig("./train_images/auto_%d.png" % epoch)
        plt.close()

    def predict_from_image(self):
        ...

    def predict_from_latent_space(self):
        ...

    def generate(self):
        ...
    

if __name__ == '__main__':

    file_paths = glob.glob("./data/*.jpg")

    images = []

    for fp in file_paths:
        img = imageio.imread(fp)
        images.append(img)

    images = np.array(images)
    print(images.shape)

    images = images / 255.0

    X_train, X_test = train_test_split(images, test_size=0.2, random_state=0)

    autoencoder = Autoencoder()
    autoencoder.train(X_train, X_test, batch_size=16, epochs=10)

    test_latent_space = autoencoder.encoder_model(X_test)
    plt.hist(test_latent_space.numpy().flatten())
    plt.show()

