from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=OPEN_API_KEY)

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a person who understands AI very well.",
    input="Explain Artificial Intelligence.",
    temperature=0.9
)

print(response.output_text)