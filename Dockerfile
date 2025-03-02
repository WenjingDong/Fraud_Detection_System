# Use AWS Lambda base image for Python
FROM public.ecr.aws/lambda/python:3.9

# Set working directory
WORKDIR /Users/wenjingdong/Personal_Projects/Fraud_Detection/fraud_detection_api

# Copy and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY config.py feature_transformer.py fraud_detection_model.pkl

# Expose the port (optional for ECS, not needed for Lambda)
EXPOSE 8080

# Lambda handler for AWS Lambda
CMD ["gunicorn", "-w 4", "-b 0.0.0.0:8080", "app:app"]
