import requests

BASE_URL = "http://localhost:4000"

def upload_dataset(file_path):
    url = f"{BASE_URL}/iris/datasets"
    with open(file_path, 'rb') as f:
        files = {'train': f}
        response = requests.post(url, files=files)
    print(response.text)
    return response.text

def create_model(dataset_id):
    url = f"{BASE_URL}/iris/model"
    data = {'dataset': dataset_id}
    response = requests.post(url, data=data)
    print(response.text)
    return response.text

def batch_test(model_id, dataset_id):
    url = f"{BASE_URL}/iris/model/{model_id}/test"
    params = {'dataset': dataset_id}
    response = requests.get(url, params=params)
    
    # Check if response was successful
    if response.status_code != 200:
        print(f"Error: Server returned status code {response.status_code}")
        print(f"Response text: {response.text}")
        return None
    
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Error: Could not decode JSON response")
        print(f"Response text: {response.text}")
        return None


if __name__ == "__main__":
    # Step 1: Upload a dataset
    dataset_id = upload_dataset("iris_extended_encoded.csv")  # Replace with your dataset file path

    # Step 2: Create a new model using the uploaded dataset
    model_id = create_model(dataset_id)

    # Step 3: Perform batch testing on the model using the same dataset
    results = batch_test(model_id, dataset_id)

    print("Batch Test Results:", results)
