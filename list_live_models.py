
import requests
import json

try:
    print("Fetching models from https://doc-ai-backend-bpez.onrender.com/api/debug/models ...")
    response = requests.get("https://doc-ai-backend-bpez.onrender.com/api/debug/models", timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        print("\nAvailable Models:")
        for m in data.get("models", []):
            if "generateContent" in m.get("methods", []):
                print(f" - {m['name']}")
    else:
        print(f"Failed: {response.status_code} - {response.text}")
        
except Exception as e:
    print(f"Error: {e}")
