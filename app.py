from flask import Flask, request, jsonify, render_template
import os

from flask_cors import CORS, cross_origin

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline


os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")


app = Flask(__name__)
CORS(app)


class ClientApp:

    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():

    os.system("python main.py")

    return "Training done successfully!"


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():

    try:

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "No image was provided."
            }), 400

        image_data = data["image"]

        if not image_data:
            return jsonify({
                "error": "Empty image data."
            }), 400

        # Save the uploaded image as inputImage.jpg
        decodeImage(image_data, clApp.filename)

        # Run prediction
        result = clApp.classifier.predict()

        return jsonify(result)

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "error": "Unable to process the image.",
            "details": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )