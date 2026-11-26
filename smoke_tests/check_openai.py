import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# Disable tracing for this isolated test
os.environ["LANGSMITH_TRACING"] = "false"

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OpenAI API key is missing.")

MODEL_NAME = "gpt-6-luna"

# Initialize the model
llm = ChatOpenAI(
    model=MODEL_NAME,
    use_responses_api=True,
    reasoning={"effort": "none"},
    max_tokens=100,
    timeout=30,
    max_retries=0
)

print("Starting OpenAI API smoke test...")
print("Model:", MODEL_NAME)

start_time = time.perf_counter()

# Execute a real API request
response = llm.invoke(
    "Reply with exactly: OK"
)

elapsed_time = time.perf_counter() - start_time

print("Response:", response.text)
print("Token usage:", response.usage_metadata)
print(f"Execution time: {elapsed_time:.2f} seconds")

assert response.text.strip() == "OK", "Unexpected response."

print("OpenAI smoke test PASSED.")