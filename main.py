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

    training = training[:10000].reshape(10000, 784)
    testing = testing[:1000].reshape(1000, 784)

    training_noise = apply_noise_to_data(training, 0.1)
    testing_noise = apply_noise_to_data(testing, 0.1)

    if os.path.exists("autoencoder.h5"):
        model = keras.models.load_model("autoencoder.h5")
        print("loaded model")
    else:
        model = create_autoencoder()
        print("created model")
        model = train_autoencoder(model, training_noise, training, testing_noise, testing, epochs=50, batch_size=16)
        print("trained model")
    model.save("autoencoder.h5")
    print("saved model")
    print("evaluating model")
    eval_pics = []
    num_pics = 100

for i in range(num_pics):
    predict_img = model.predict(x=np.array([testing_noise[i]]))
    predict_img = denormalize(data=predict_img)
    test_img = denormalize(data=np.array([testing_noise[i]]))
    cv2.imwrite("eval/" + str(i) + "_predict.jpg",
                predict_img.reshape(28, 28))
    cv2.imwrite("eval/" + str(i) + ".jpg",
                test_img.reshape(28, 28))
