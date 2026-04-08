
import requests
import json
import time

BASE_URL = "https://doc-ai-backend-bpez.onrender.com/api"

def run_test():
    print(f"Testing Remote Server: {BASE_URL}")
    
    # 1. Check Config
    try:
        r = requests.get(f"{BASE_URL}/config", timeout=10)
        print(f"Config Status: {r.status_code}")
        if r.status_code == 200:
            print(f"Config: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"Config check failed: {e}")

    # 2. Upload Document
    print("\nUploading test document...")
    files = {
        'file': ('test_doc.txt', 'Shipper: ACME Corp\nConsignee: RoadRunner Inc\nWeight: 1000 lbs\nDate: 2026-02-11', 'text/plain')
    }
    
    try:
        r = requests.post(f"{BASE_URL}/upload", files=files, timeout=30)
        if r.status_code != 200:
            print(f"Upload Failed: {r.text}")
            return
            
        data = r.json()
        doc_id = data['document_id']
        print(f"Upload Success! ID: {doc_id}")
        
    except Exception as e:
        print(f"Upload Error: {e}")
        return

    # Wait for indexing
    time.sleep(2)

    # 3. Test Extraction
    print(f"\nTesting Extraction for {doc_id}...")
    try:
        r = requests.post(f"{BASE_URL}/extract", json={"document_id": doc_id}, timeout=30)
        if r.status_code == 200:
            ext = r.json()
            print(f"Extraction Result: {json.dumps(ext, indent=2)}")
        else:
            print(f"Extraction Failed: {r.text}")
    except Exception as e:
        print(f"Extraction Error: {e}")

    # 4. Test Chat
    print(f"\nTesting Chat for {doc_id}...")
    try:
        r = requests.post(f"{BASE_URL}/ask", json={"document_ids": [doc_id], "question": "Who is the shipper?"}, timeout=30)
        if r.status_code == 200:
            ans = r.json()
            print(f"Chat Answer: {ans['answer']}")
            print(f"Confidence: {ans.get('confidence')}")
        else:
            print(f"Chat Failed: {r.text}")
    except Exception as e:
        print(f"Chat Error: {e}")

    # 5. Cleanup
    print(f"\nCleaning up {doc_id}...")
    requests.delete(f"{BASE_URL}/documents/{doc_id}")

if __name__ == "__main__":
    run_test()
