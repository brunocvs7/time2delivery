import requests
import sys
import json

import pandas as pd

# Getting filename as argument
filename_input = sys.argv[1]
filename_output = sys.argv[2]
# Reading parquet file
X = pd.read_parquet(filename_input)
# Fillna (We cannot jsonify with missing)
X.fillna(9999, inplace=True)
# Transforming datetime to string (We cannot jsonify with datetime)
X['promised_time'] = X['promised_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# list to save each json generated 
list_predictions = []
# Transforming records of dataframe into list of dictionaries
X_list = X.to_dict(orient='records')

# for each item of the list We will make a request 
for records in X_list[0:10]:
    res = requests.get('http://localhost:5000/predict', json=records)
    list_predictions.append(res.json())
    
# Exporting
json_response = pd.DataFrame.from_records(list_predictions)
json_response.to_csv(f'data/predictions/{filename_output}', index=False)