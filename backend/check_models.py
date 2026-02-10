
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from dotenv import load_dotenv
load_dotenv("backend/.env")

from backend.config import get_settings
import google.generativeai as genai

settings = get_settings()

def list_models():
    print(f"Checking models with key: {settings.gemini_api_key[:5]}...")
    genai.configure(api_key=settings.gemini_api_key)
    
    try:
        print("\nAvailable Models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
