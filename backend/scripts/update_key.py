import os
import sys

def update_key(new_key):
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(backend_dir, ".env")
    
    if not os.path.exists(env_path):
        print(f"Error: .env file not found at {env_path}")
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        key_updated = False
        
        for line in lines:
            if line.startswith("GEMINI_API_KEY="):
                new_lines.append(f"GEMINI_API_KEY={new_key}\n")
                key_updated = True
            else:
                new_lines.append(line)
        
        if not key_updated:
            new_lines.append(f"\nGEMINI_API_KEY={new_key}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
            
        print(f"Successfully updated GEMINI_API_KEY in {env_path}")
        print("Please restart the backend for changes to take effect.")

    except Exception as e:
        print(f"Failed to update key: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_key.py <NEW_API_KEY>")
    else:
        update_key(sys.argv[1])
