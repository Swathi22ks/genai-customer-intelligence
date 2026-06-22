"""
sagemaker_deploy.py
===================
Deploy the churn model to AWS SageMaker.

Skills: AWS, SageMaker, S3, cloud deployment, model serving
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

AWS_REGION       = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
S3_BUCKET        = os.getenv("AWS_S3_BUCKET", "genai-customer-intelligence")
MODEL_PATH       = "models/churn_model.pt"
ENDPOINT_NAME    = "genai-churn-predictor-v1"
INSTANCE_TYPE    = "ml.t2.medium"  # Use ml.g4dn.xlarge for GPU


def upload_model_to_s3(local_path: str, bucket: str, key: str) -> str:
    """Upload trained model artifact to S3."""
    s3 = boto3.client("s3", region_name=AWS_REGION)

    logger.info(f"[AWS] Uploading model to s3://{bucket}/{key}")
    s3.upload_file(local_path, bucket, key)

    s3_uri = f"s3://{bucket}/{key}"
    logger.info(f"[AWS] Model uploaded: {s3_uri}")
    return s3_uri


def create_sagemaker_endpoint(model_s3_uri: str,
                               endpoint_name: str = ENDPOINT_NAME) -> str:
    """Create a SageMaker real-time inference endpoint."""
    session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))
    role = sagemaker.get_execution_role()  # Only works inside SageMaker Studio

    pytorch_model = PyTorchModel(
        model_data=model_s3_uri,
        role=role,
        framework_version="2.1",
        py_version="py310",
        entry_point="inference.py",
        source_dir="cloud/aws/",
        env={
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "60",
        },
        name="genai-churn-model",
    )

    logger.info(f"[AWS] Deploying endpoint: {endpoint_name}...")
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=endpoint_name,
        wait=True,
    )

    logger.info(f"[AWS] Endpoint deployed: {endpoint_name} ✓")
    return endpoint_name


def invoke_endpoint(endpoint_name: str, payload: dict) -> dict:
    """
    Invoke a deployed SageMaker endpoint for real-time prediction.

    Args:
        endpoint_name: Name of the SageMaker endpoint
        payload: Dict with "features" key (list of floats)

    Returns:
        Dict with "churn_probability" and "predicted_class"
    """
    import json
    client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read())
    return result


def delete_endpoint(endpoint_name: str):
    """Delete endpoint to avoid unnecessary charges."""
    client = boto3.client("sagemaker", region_name=AWS_REGION)
    client.delete_endpoint(EndpointName=endpoint_name)
    logger.info(f"[AWS] Endpoint deleted: {endpoint_name}")


# ──────────────────────────────────────────────────────────────
# INFERENCE HANDLER (uploaded to SageMaker with model)
# ──────────────────────────────────────────────────────────────

INFERENCE_SCRIPT = '''
"""
inference.py — SageMaker inference handler for ChurnNet
"""
import torch
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def model_fn(model_dir):
    """Load model from model_dir (called once on startup)."""
    from src.deep_learning.churn_model import ChurnNet
    checkpoint = torch.load(
        os.path.join(model_dir, "churn_model.pt"),
        map_location="cpu"
    )
    model = ChurnNet(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        features = np.array(data["features"], dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return torch.FloatTensor(features)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Run inference."""
    with torch.no_grad():
        prob = model(input_data).numpy()
    return prob


def output_fn(prediction, accept):
    """Format output."""
    result = {
        "churn_probability": float(prediction[0]),
        "predicted_class": int(prediction[0] >= 0.5),
        "risk_level": "High" if prediction[0] >= 0.7 else "Medium" if prediction[0] >= 0.4 else "Low",
    }
    return json.dumps(result), "application/json"
'''


def write_inference_script():
    """Write the inference handler script."""
    with open("cloud/aws/inference.py", "w") as f:
        f.write(INFERENCE_SCRIPT)
    logger.info("[AWS] inference.py written for SageMaker deployment.")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy churn model to AWS SageMaker")
    parser.add_argument("--action", choices=["deploy", "delete", "test"],
                        default="deploy", help="Action to perform")
    args = parser.parse_args()

    if args.action == "deploy":
        write_inference_script()
        model_uri = upload_model_to_s3(
            local_path=MODEL_PATH,
            bucket=S3_BUCKET,
            key="models/churn_model.pt",
        )
        create_sagemaker_endpoint(model_uri)

    elif args.action == "delete":
        delete_endpoint(ENDPOINT_NAME)

    elif args.action == "test":
        # Test endpoint with dummy features (21 features)
        dummy_features = [0.5] * 21
        result = invoke_endpoint(ENDPOINT_NAME, {"features": dummy_features})
        print("Prediction result:", result)
