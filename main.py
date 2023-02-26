import keras.models
from Autoencoder import *
from helpers import *
import os
if __name__ == "__main__":
    # Load data
    training = load_images(True)
    testing = load_images(False)

    training = normalize(data=training)
    testing = normalize(data=testing)

    training_noise = apply_noise_to_data(training, 0.75)
    testing_noise = apply_noise_to_data(testing, 0.75)

    if os.path.exists("autoencoder.h5"):
        model = keras.models.load_model("autoencoder.h5")
        print("loaded model")
    else:
        model = create_autoencoder()
        print("created model")
    model = train_autoencoder(
        model, training_noise, training, testing_noise, testing, epochs=250, batch_size=32)
    print("trained model")
    model.save("autoencoder.h5")
    print("saved model")
    print("evaluating model")
    eval_pics = []
    num_pics = 100

for i in range(num_pics):
    predict_img = model.predict(x=np.array([testing_noise[i]]))
    predict_img = denormalize(data=predict_img)
    test_img = denormalize(data=np.array([testing[i]]))
    cv2.imwrite("eval/" + str(i) + "_predict.jpg",
                predict_img.reshape(28, 28))
    cv2.imwrite("eval/" + str(i) + ".jpg",
                test_img.reshape(28, 28))
