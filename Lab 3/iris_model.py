from flask import Flask, request, jsonify
from base_iris_lab1 import load_local, build, train, score, new_model, add_dataset, datasets, models
import pandas as pd
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np

app = Flask(__name__)

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

@app.route('/iris/model/<int:model_id>/test', methods=['POST'])
def test_model(model_id):
    if 'dataset' not in request.form:
        return "Dataset index required", 400
    
    try:
        dataset_id = int(request.form['dataset'])
        if dataset_id >= len(datasets) or model_id >= len(models):
            return "Invalid dataset or model ID", 400
        
        # Get dataset and model
        dataset = datasets[dataset_id]
        model = models[model_id]
        
        # Prepare data for evaluation
        X = dataset.iloc[:, 1:].values  # Features (excluding species)
        y = dataset.iloc[:, 0].values   # Species
        encoder = LabelEncoder()
        y1 = encoder.fit_transform(y)
        Y = pd.get_dummies(y1).values
        
        # Evaluate model
        loss, accuracy = model.evaluate(X, Y, verbose=0)
        y_pred = model.predict(X, verbose=0)
        actual = np.argmax(Y, axis=1)
        predicted = np.argmax(y_pred, axis=1)
        
        # Compute metrics
        conf_matrix = confusion_matrix(actual, predicted)
        precision = precision_score(actual, predicted, average=None).tolist()
        recall = recall_score(actual, predicted, average=None).tolist()
        
        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'precision': precision,
            'recall': recall
        }
        
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Removed this line to prevent loading a default dataset
    # load_local()
    app.run(host='0.0.0.0', port=5000, debug=True)