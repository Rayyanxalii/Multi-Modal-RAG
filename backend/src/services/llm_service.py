import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()

llm = Groq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API"),
)

llm_vision = Groq(
    model="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API"),
    max_tokens=2048,
)