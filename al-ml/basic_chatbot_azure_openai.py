#!/usr/bin/env python3
"""
Basic ChatBot with Azure OpenAI, packaged as an MLflow pyfunc model.

This script mirrors the functionality of the original Databricks notebook:
- Creates an Azure OpenAI client
- Calls Chat Completions
- Wraps the logic in an MLflow `pyfunc.PythonModel`
- Saves, loads, tests, and logs the model to MLflow

Prerequisites:
- pip install openai==1.56.0 mlflow pandas
- Set environment variables:
    AZURE_OPENAI_ENDPOINT=<your-endpoint, e.g. https://<resourcename>.openai.azure.com/>
    AZURE_OPENAI_API_KEY=<your-api-key>
    AZURE_OPENAI_API_VERSION=2024-02-15-preview  (optional; defaults to this value)
    AZURE_OPENAI_DEPLOYMENT_NAME=<your-deployment/model name>
"""

import os
from typing import Any

import pandas as pd
import mlflow
from mlflow import pyfunc
from mlflow.models import infer_signature
from openai import AzureOpenAI


# -----------------------------
# Azure OpenAI Client Helpers
# -----------------------------

def _get_azure_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()

    if not endpoint or not api_key:
        raise RuntimeError(
            "Missing required environment variables AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY."
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def chat_once(system_prompt: str, user_prompt: str, model: str, temperature: float = 0.7) -> str:
    client = _get_azure_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# -----------------------------
# MLflow pyfunc Model
# -----------------------------

class BasicChatBot(pyfunc.PythonModel):
    """
    Minimal chat-bot model wrapping Azure OpenAI Chat Completions API.
    Expects a DataFrame with a 'user_query' column; returns a string response.
    """

    def __init__(self, gpt_model: str):
        self.gpt_model = gpt_model

    def chat_completions_api(self, user_query: str) -> str:
        client = _get_azure_client()
        response = client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": f"{user_query}"},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

    def predict(self, context: Any, data: pd.DataFrame) -> Any:
        if "user_query" not in data.columns:
            raise ValueError("Input DataFrame must contain a 'user_query' column.")
        # For simplicity, handle the first row only, mirroring the notebook's behavior
        user_query = data["user_query"].iloc[0]
        return self.chat_completions_api(user_query)


# -----------------------------
# Main flow (save, load, test, log)
# -----------------------------

def main():
    # Pull the Azure OpenAI deployment/model name from env, with placeholder fallback.
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "YOUR-MODEL-NAME")

    # Quick direct chat call (equivalent to the notebook's single call block)
    system_prompt = "You are a helpful AI assistant meant to answer the user query"
    user_prompt = "Hi, how are you?"
    try:
        print("=== Direct Chat Call ===")
        direct_resp = chat_once(system_prompt, user_prompt, model=deployment_name, temperature=0.7)
        print(direct_resp)
        print()
    except Exception as e:
        print(f"Direct chat call failed: {e}\n(Continuing with model creation...)")

    # Create and save the MLflow model
    print("=== Saving MLflow pyfunc model ===")
    test_model = BasicChatBot(gpt_model=deployment_name)

    # Define input signature (DataFrame with a single 'user_query' string column)
    example_input = pd.DataFrame([{"user_query": "hello how are you?"}])
    signature = infer_signature(example_input)

    model_path = "basicchatbot"
    mlflow.pyfunc.save_model(path=model_path, python_model=test_model, signature=signature)
    print(f"Model saved to: {model_path}\n")

    # Load and test the saved model
    print("=== Loading and testing saved model ===")
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_path)
    model_input = pd.DataFrame([{"user_query": "hello how are you?"}])

    try:
        model_response = loaded_pyfunc_model.predict(model_input)
        print("Model response:", model_response)
    except Exception as e:
        print(f"Model prediction failed: {e}")
    print()

    # Log model artifacts to MLflow
    print("=== Logging model artifacts to MLflow ===")
    with mlflow.start_run() as run:
        mlflow.log_artifacts(local_dir=model_path, artifact_path="BasicChatBot")
        print(f"Model logged with run ID: {run.info.run_id}\n")

    # Example of real-time inference payload (commented for reference)
    print("=== Example real-time inference payload (JSON) ===")
    print(
        '{\n'
        '  "dataframe_records": [\n'
        '    {\n'
        '      "user_query": "tell me something about India"\n'
        '    }\n'
        '  ]\n'
        '}'
    )


if __name__ == "__main__":
    main()
