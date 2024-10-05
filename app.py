from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS
import logging


app = Flask(__name__)
CORS(app, resources={"/predict": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG

# Load the logistic regression model
try:
    # lr = joblib.load("model/logistic_regression_model.pkl")
    # DecisionTree = joblib.load("model/decision_tree_model.pkl")
    lgbm = joblib.load("model/lightgbm_model.pkl")

except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")


def preprocess_input(data):
    try:
        # Convert input data to numpy array and reshape
        data_array = np.array(data).reshape(1, -1)
        app.logger.debug("This is a debug message")

        return data_array
    except Exception as e:
        raise ValueError(f"Error in data preprocessing: {e}")


@app.route("/", methods=["GET"])
def home():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract data from the POST request
        json_data = request.json
        print(f"Received JSON data: {json_data}")
        if "data" not in json_data:
            return jsonify({"error": "Invalid input format, 'data' key not found"}), 400

        data = json_data["data"]
        if not isinstance(data, list) or not all(
            isinstance(i, (int, float)) for i in data
        ):
            return (
                jsonify(
                    {
                        "error": "Invalid input format, expected a list of numbers under 'data'"
                    }
                ),
                400,
            )
        print(f"Received data: {data}")
        # Preprocess the data

        preprocessed_data = preprocess_input(data)
        print(f"processed data: {preprocessed_data}")
        # Predict using the logistic regression model
        lgbm_pred = lgbm.predict(preprocessed_data)

        # DecisionTree_pred = DecisionTree.predict(preprocessed_data)

        # Return the prediction as a JSON response
        predictions = {
            "result": str(lgbm_pred[0]),
            # "DT Prediction": int(DecisionTree_pred[0]),
        }
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
