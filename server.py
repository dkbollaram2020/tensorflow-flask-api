from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

__model = None
__scaler = None

def load_saved_artifacts():
    global __model
    global __scaler

    __model = load_model("final_model.h5")
    __scaler = joblib.load("scaler.pkl")

@app.route('/')
def get_home():
    return jsonify({"message": "server works.."})

@app.route('/predict_home_price')
def predict_home_price():
    # try:
    square_footage = float(request.args.get('square_footage'))
    bedrooms = int(request.args.get('bedrooms'))
    bathrooms = int(request.args.get('bathrooms'))
    sfr = int(request.args.get('sfr'))
    print(square_footage)
    print(bedrooms)
    print(bathrooms)
    print(sfr)
    results = return_prediction(bedrooms,bathrooms,square_footage,sfr, model=__model,scaler=__scaler)       
    return jsonify(results)
    # except:
    #     return jsonify({"message": "Error occurred"})


def return_prediction(bedrooms,bathrooms,square_footage,sfr,model,scaler):
    new_home = [[bedrooms,bathrooms,square_footage,sfr]]
    print(new_home)
    # new_home = __scaler.transform(new_home)   
    new_home = [[0.4, 0.33333333, 0.39338655, 0.0]]  
    print(new_home)
    result = __model.predict(new_home)    
    return result

if __name__ == "__main__":
    print("Starting server...")
    load_saved_artifacts()
    app.run()
