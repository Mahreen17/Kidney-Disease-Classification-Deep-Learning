import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        # Load the trained model
        model = load_model("artifacts/training/model.h5")

        # Load image and resize it to VGG16 input size
        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        # Convert image to NumPy array
        test_image = image.img_to_array(test_image)

        # Apply the SAME preprocessing used for the VGG16 model
        test_image = preprocess_input(test_image)

        # Add batch dimension
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        prediction_probability = model.predict(
            test_image,
            verbose=0
        )

        # Get predicted class
        result = np.argmax(prediction_probability, axis=1)

        # Class mapping:
        # Normal = 0
        # Tumor = 1
        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        print("Prediction probabilities:", prediction_probability)
        print("Predicted class:", result[0])
        print("Prediction:", prediction)

        return [{"image": prediction}]