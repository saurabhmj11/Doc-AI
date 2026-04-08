
import requests
import time
import sys

URL = "https://doc-ai-backend-bpez.onrender.com/api/debug/models"

def monitor():
    print(f"📡 Monitoring deployment at: {URL}")
    print("Waiting for new version with debug route to go live...")
    
    while True:
        try:
            response = requests.get(URL, timeout=10)
            status = response.status_code
            
            if status == 200:
                print("\n✅ DEPLOYMENT LIVE!")
                print("New route /api/debug/models is active.")
                print("Response:", response.json())
                break
            elif status == 404:
                sys.stdout.write(".")
                sys.stdout.flush()
            else:
                print(f"\n⚠️ Status {status}")
                
        except Exception as e:
            print(f"\n❌ Connection error: {e}")
            
        time.sleep(10)

if __name__ == "__main__":
    monitor()
