FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements-docker.txt .

# Install any needed packages specified in requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Install a compatible version of TensorFlow separately (check documentation for compatible version)
RUN pip install tensorflow

# Copy the rest of the application code into the container
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME .env

# Run app.py when the container launches
CMD ["python", "app.py"]
