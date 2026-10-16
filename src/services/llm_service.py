import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()

llm = Groq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API"),
)

response = llm.complete("Explain RAG in simple terms.")
print(response)