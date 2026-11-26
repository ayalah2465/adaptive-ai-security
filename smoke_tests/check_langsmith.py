import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_openai import ChatOpenAI
from langsmith import Client

# Validate environment configuration
for key in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
    if not os.getenv(key):
        raise RuntimeError(f"Missing environment variable: {key}")

if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
    raise RuntimeError("LangSmith tracing is not enabled.")

MODEL_NAME = "gpt-6-luna"
PROJECT_NAME = os.environ["LANGSMITH_PROJECT"]

print("Starting LangSmith smoke test...")
print("Project:", PROJECT_NAME)
print("Model:", MODEL_NAME)

# Verify LangSmith API access
client = Client()
projects = list(client.list_projects(limit=1))

print("LangSmith API connection: OK")

if not projects:
    print("No existing tracing projects found.")
    print("This is normal for a new workspace.")

# Initialize the model
llm = ChatOpenAI(
    model=MODEL_NAME,
    use_responses_api=True,
    reasoning={"effort": "none"},
    max_tokens=100,
    timeout=30,
    max_retries=0
)

start_time = time.perf_counter()

# Execute a traced model request
response = llm.invoke(
    "Reply with exactly: OK",
    config={
        "run_name": "LangSmith Environment Smoke Test",
        "tags": ["smoke-test", "environment-validation"]
    }
)

elapsed_time = time.perf_counter() - start_time

print("Response:", response.text)
print("Token usage:", response.usage_metadata)
print(f"Execution time: {elapsed_time:.2f} seconds")

assert response.text.strip() == "OK", "Unexpected response."

print("Model invocation PASSED.")
print("Next step: verify the trace in LangSmith.")