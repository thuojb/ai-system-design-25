import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
from flask.json import JSONEncoder
from flask import Flask, request, jsonify
from base_iris_lab1 import load_local, build, train, score, new_model, add_dataset
import pandas as pd
import io
from test import test

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

@app.route('/iris/datasets', methods=['POST'])
def create_dataset():
    if 'train' not in request.files:
        return "No file part", 400
    
    file = request.files['train']
    if file.filename == '':
        return "No selected file", 400
    
    # Read CSV file from the request
    csv_data = file.read()
    df = pd.read_csv(io.StringIO(csv_data.decode('utf-8')))
    
    # Add dataset to the model service
    dataset_id = add_dataset(df)
    
    return str(dataset_id), 200

@app.route('/iris/model', methods=['POST'])
def create_model():
    if 'dataset' not in request.form:
        return "Dataset index required", 400
    
    dataset_id = int(request.form['dataset'])
    
    # Create and train a new model
    model_id, history = new_model(dataset_id)
    
    return str(model_id), 200

@app.route('/iris/model/<int:model_id>', methods=['PUT'])
def retrain_model(model_id):
    if 'dataset' not in request.args:
        return "Dataset index required", 400
    
    dataset_id = int(request.args.get('dataset'))
    
    # Train the existing model with the specified dataset
    history = train(model_id, dataset_id)
    
    return str(history), 200

@app.route('/iris/model/<int:model_id>/score', methods=['GET'])
def score_model(model_id):
    if 'fields' not in request.args:
        return "Fields required", 400
    
    # Parse the fields parameter (comma-separated values)
    fields_str = request.args.get('fields')
    fields = [float(f) for f in fields_str.split(',')]
    
    # Score the model with the provided fields
    result = score(model_id, fields)
    
    return result

@app.route('/iris/model/<int:model_id>/test', methods=['GET'])
def batch_test(model_id):
    if 'dataset' not in request.args:
        return "Dataset ID required", 400

    dataset_id = int(request.args.get('dataset'))

    try:
        results = test(model_id, dataset_id)
        return jsonify(results), 200
    except Exception as e:
        return f"An error occurred: {str(e)}", 500



if __name__ == '__main__':
    # Remove this line to prevent loading a default dataset
    # load_local()
    app.run(host='0.0.0.0', port=4000, debug=True)