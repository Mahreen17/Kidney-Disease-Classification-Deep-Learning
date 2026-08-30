import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

        self.interpreter = tf.lite.Interpreter(
            model_path="model/model.tflite"
        )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self):

        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        test_image = image.img_to_array(test_image)

        test_image = preprocess_input(test_image)

        test_image = np.expand_dims(test_image, axis=0)

        test_image = test_image.astype(
            self.input_details[0]["dtype"]
        )

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            test_image
        )

        self.interpreter.invoke()

        prediction_probability = self.interpreter.get_tensor(
            self.output_details[0]["index"]
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

        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{"image": prediction}]