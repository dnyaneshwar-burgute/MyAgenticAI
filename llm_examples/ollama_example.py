# Command to run ollama
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model=OLLAMA_MODEL,
    messages=[
        {"role": "user",
         "content": "Explain Artificial Intelligence.",
        },
        
    ]
)

print(response.choices[0].message.content)