from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)

__model = None

def load_saved_artifacts():
    global __model
    __model = load_model("model_with_resident_type.h5")
    print("Model Loaded...")


@app.route('/')
def get_home():
    return jsonify({"message": "server works...!"})

@app.route('/predict_home_price')
def predict_home_price():
    square_footage = float(request.args.get('square_footage'))
    bedrooms = int(request.args.get('bedrooms'))
    bathrooms = int(request.args.get('bathrooms'))
    sfr = int(request.args.get('sfr'))
    results = return_prediction(bedrooms,bathrooms,square_footage,sfr, model=__model)       
    return jsonify({"Predicted Price": round(results[0], 2)})



def return_prediction(bedrooms,bathrooms,square_footage,sfr,model):
    # new_home = [[bedrooms,bathrooms,square_footage,sfr]]
    new_home = np.array([[bedrooms,bathrooms,square_footage,sfr]])
    result = __model.predict(new_home)
    return result[0].tolist()

if __name__ == "__main__":
    print("Starting server...")
    load_saved_artifacts()
    app.run()
