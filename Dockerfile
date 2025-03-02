# Use AWS Lambda base image for Python
# FROM public.ecr.aws/lambda/python:3.9

# Not using AWS
FROM python:3.9

# Set working directory
WORKDIR /Users/wenjingdong/Personal_Projects/Fraud_Detection/fraud_detection_api

# Copy and install dependencies
COPY requirements.txt .

# RUN pip install -r requirements.txt
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the rest of the application code
COPY config.py  fraud_detection_model_fixed.pkl app.py .
COPY feature_transformer.py .
COPY templates/ ./templates/

# Expose the port (optional for ECS, not needed for Lambda)
EXPOSE 8080

# Remove AWS Lambda entrypoint (if it exists)
ENTRYPOINT []

# Lambda handler for AWS Lambda

# Flask API
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
