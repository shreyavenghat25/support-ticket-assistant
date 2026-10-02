import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key, model = os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL")
if not key or not model:
    raise SystemExit("Missing GEMINI_API_KEY or GEMINI_MODEL in .env")

client = genai.Client(api_key=key)
reply = client.models.generate_content(
    model=model,
    contents="In one short sentence: which support department should handle "
             "'I was charged twice for my subscription this month'?",
)
print(f"Model: {model}")
print(f"Reply: {reply.text}")
