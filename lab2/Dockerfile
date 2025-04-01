# Use TensorFlow as the base image
FROM tensorflow/tensorflow:latest-py3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Make port 4000 available
EXPOSE 4000

# Run the Flask application
CMD ["python", "iris_model.py"]
