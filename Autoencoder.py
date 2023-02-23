from keras.layers import Input, Dense
from keras.models import Model


def create_autoencoder():
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation="relu")(input_img)
    encoded = Dense(64, activation="relu")(encoded)
    encoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dense(784, activation="linear")(decoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    return autoencoder


def train_autoencoder(autoencoder, x_train, x_train_noise, x_test, x_test_noise, epochs=1000, batch_size=128):
    autoencoder.fit(x_train, x_train_noise,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test_noise))
    return autoencoder
