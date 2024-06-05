# Use the official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements-docker.txt .

# Install any needed packages specified in requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Install TensorFlow if necessary, or a compatible version
RUN pip install tensorflow

# Copy the rest of the application code into the container
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variables if needed (optional)
# ENV VAR_NAME value

# Run app.py when the container launches
CMD ["python", "app.py"]
