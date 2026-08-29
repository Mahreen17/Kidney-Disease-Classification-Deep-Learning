import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        # Load the trained model from the model folder
        model = load_model("model/model.h5")

        # Load image
        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        # Convert image to array
        test_image = image.img_to_array(test_image)

        # Normalize exactly like training
        test_image = test_image / 255.0

        # Add batch dimension
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        prediction_probability = model.predict(
            test_image,
            verbose=0
        )

        result = np.argmax(
            prediction_probability,
            axis=1
        )

        print(
            "Prediction probabilities:",
            prediction_probability
        )

        print(
            "Predicted class:",
            result[0]
        )

        # Class mapping:
        # Normal = 0
        # Tumor = 1

        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{"image": prediction}]