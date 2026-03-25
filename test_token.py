from dotenv import load_dotenv   #type: ignore
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"Token found: {token is not None}")
print(f"Token value: '{token}'")
print(f"Token length: {len(token) if token else 0}")