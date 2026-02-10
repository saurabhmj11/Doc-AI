
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings

def check_key():
    try:
        settings = get_settings()
        key = settings.gemini_api_key
        if not key:
            print("API Key Status: MISSING (Empty String)")
        elif key == "your_api_key_here":
            print("API Key Status: INVALID (Default/Example Value)")
        else:
            masked = key[:4] + "***" + key[-4:] if len(key) > 8 else "***"
            print(f"API Key Status: PRESENT ({masked})")
    except Exception as e:
        print(f"Error loading settings: {e}")

if __name__ == "__main__":
    check_key()
