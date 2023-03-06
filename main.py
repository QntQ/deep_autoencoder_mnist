import keras.models
from Autoencoder import *
from helpers import *
import os
import keras.datasets.fashion_mnist
if __name__ == "__main__":
    # Load data
    # training = load_images(True)
    # testing = load_images(False)
    (training, _), (testing, _) = keras.datasets.fashion_mnist.load_data()
    training = normalize(data=training)
    testing = normalize(data=testing)

    training_noise = add_artifact_to_data(training)
    testing_noise = add_artifact_to_data(testing)

    training = training[:60000].reshape(60000, 784)
    testing = testing[:10000].reshape(10000, 784)

    training_noise = training_noise[:60000].reshape(60000, 784)
    testing_noise = testing_noise[:10000].reshape(10000, 784)

    if os.path.exists("autoencoder.h5"):
        model = keras.models.load_model("autoencoder.h5")
        print("loaded model")
    else:
        model = create_autoencoder()
        print("created model")
        model = train_autoencoder(
            model, training_noise, training, testing_noise, testing, epochs=200, batch_size=16)
        print("trained model")
    model.save("autoencoder.h5")
    print("saved model")
    print("evaluating model")

    evaluate(model, testing, testing_noise, num_output=100)
