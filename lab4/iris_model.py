from flask import Flask, request, jsonify
from base_iris_lab1 import load_local, build, train, score, new_model, add_dataset, datasets, models
import pandas as pd
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
from datetime import datetime

# Import AWS logging dependencies
from lab4_header import scores_table
from post_score import post_score

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
        
        # Map encoded labels back to class names
        class_names = encoder.classes_  # e.g., ['setosa', 'versicolor', 'virginica']
        
        # Log each test record to DynamoDB
        for i in range(len(X)):
            # Features as a comma-separated string
            # feature_string = ','.join(map(str, X[i]))
            feature_cols = dataset.columns[1:].tolist()  # Get feature column names (e.g., ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            feature_dict = {feature_cols[j]: float(X[i][j]) for j in range(len(feature_cols))}
            feature_dict['species'] = y[i]  # Add original species value
            feature_string = str(feature_dict).replace("'", '"')
            
            # Predicted class
            pred_class = class_names[predicted[i]]
            
            # Actual class
            actual_class = class_names[actual[i]]
            
            # Prediction probability (max probability for the predicted class)
            prob = float(y_pred[i][predicted[i]])
            prob_string = f"{prob:.3f}"
            
            # Write to DynamoDB using post_score
            response = post_score(
                log_table=scores_table,
                feature_string=feature_string,
                class_string=pred_class,
                actual_string=actual_class,
                prob_string=prob_string
            )
            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                print(f"Failed to log record {i}: {response}")
        
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
    app.run(host='0.0.0.0', port=5000, debug=True)