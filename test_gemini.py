import os
from dotenv import load_dotenv
from google import genai

print("Starting Gemini connection test...", flush=True)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: API key not found in .env")
    raise SystemExit(1)

print("API key loaded successfully.", flush=True)

try:
    client = genai.Client(api_key=api_key)

    print("Sending request to Gemini...", flush=True)

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Hello Gemini! Introduce yourself in 3 sentences."
    )

    print("\n========== GEMINI RESPONSE ==========")
    print(response.text)
    print("=====================================")

except Exception as error:
    print("\nGEMINI API ERROR:")
    print(type(error).__name__, error)

finally:
    if "client" in locals():
        client.close()