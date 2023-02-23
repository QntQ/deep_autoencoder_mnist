from tensorflow.keras.utils import plot_model  # for plotting model diagram
# for adding layers to DAE model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Dense
from keras import Input  # for instantiating a keras tensor
from keras.models import Model  # for creating a Neural Network Autoencoder model


from tensorflow import keras  # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__)  # print version


def create_autoencoder():

    n_inputs = 784
    # --- Input Layer
    # Specify input shape
    visible = Input(shape=(n_inputs,), name='Input-Layer')

    # --- Encoder Layer
    e = Dense(units=n_inputs, name='Encoder-Layer')(visible)
    e = BatchNormalization(name='Encoder-Layer-Normalization')(e)
    e = LeakyReLU(name='Encoder-Layer-Activation')(e)

    # --- Middle Layer
    middle = Dense(units=n_inputs, activation='linear', activity_regularizer=keras.regularizers.L1(
        0.0001), name='Middle-Hidden-Layer')(e)

    # --- Decoder Layer
    d = Dense(units=n_inputs, name='Decoder-Layer')(middle)
    d = BatchNormalization(name='Decoder-Layer-Normalization')(d)
    d = LeakyReLU(name='Decoder-Layer-Activation')(d)

    # --- Output layer
    output = Dense(units=n_inputs, activation='sigmoid',
                   name='Output-Layer')(d)

    # Define denoising autoencoder model
    model = Model(inputs=visible, outputs=output,
                  name='Denoising-Autoencoder-Model')

    # Compile denoising autoencoder model
    model.compile(optimizer='adam', loss='mse')
    return model


def train_autoencoder(autoencoder, x_train_noise, x_train, x_test_noise, x_test, epochs=1000, batch_size=128):
    autoencoder.fit(x_train_noise, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test_noise))
    return autoencoder
