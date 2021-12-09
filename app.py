import yaml
import pickle
import pandas as pd
import json 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from yaml.loader import SafeLoader
from utils.models.predict_model import get_intervals
from datetime import datetime

# constants
HOST = '0.0.0.0'
PORT = 5000

# configs
with open('config/config_model.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)
    all_columns = config['columns']
    model_path = config['model_path']
    feature_engineering_pipe_path = config['feature_enginering_pipe_path']

# Load Model
model = pickle.load(open(model_path, 'rb'))
# Feature Engineering Pipeline
feature_engineering_pipe = pickle.load(open(feature_engineering_pipe_path, 'rb'))

# initialize flask application
app = Flask(__name__)

@app.route('/')
def home():
    return 'Time to Delivery is Running'


@app.route('/predict', methods=['GET'])
def predict():
    X = request.get_json(force=True)
    X = pd.DataFrame(X, index=[0])
    X.replace({999:np.nan})
    X['promised_time'] = pd.to_datetime(X['promised_time'],format='%Y-%m-%d %H:%M:%S', errors='coerce')
    X = feature_engineering_pipe.transform(X)
    order_id = X['order_id'][0]
    X.drop('order_id', inplace=True, axis=1)
    X = X.loc[:,all_columns]
    y_prediction =  model.predict(X)[0]
    dist_prediction = get_intervals(model=model, X=X, confidence=0.95)
    lower_bound_prediction = dist_prediction['interval'][0][0]
    upper_bound_prediction = dist_prediction['interval'][0][1]
    results = {'order_id':order_id, 'total_minutes':y_prediction, 'lower_bound_95':lower_bound_prediction, 'upper_bound_95':upper_bound_prediction}
    return json.dumps(results)

if __name__ == '__main__':
    # run web server
    app.run(host=HOST, 
            debug=True,  
            port=PORT)
