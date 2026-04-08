
import os
import sys
import google.generativeai as genai

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_settings

try:
    settings = get_settings()
    print(f"--- Configuration Verification ---")
    
    # Check LLM Mode
    print(f"LLM_MODE: {settings.llm_mode}")
    if settings.llm_mode != "online":
        print("WARNING: LLM_MODE is NOT 'online'. Gemini API will NOT be used.")
        print(f"Current mode: {settings.llm_mode}")
    else:
        print("Success: LLM_MODE is 'online' (Gemini enabled)")

    print(f"\n--- API Key Validation ---")
    
    if not settings.gemini_api_key:
        print("ERROR: API Key is missing in settings.")
        sys.exit(1)

    print(f"Key present (Length: {len(settings.gemini_api_key)})")
    
    genai.configure(api_key=settings.gemini_api_key)
    
    print("Attempting to list models...")
    try:
        models = list(genai.list_models())
        print(f"Success! Found {len(models)} models.")
        
        # Check for configured models
        configured_gen_model = settings.gemini_model
        
        found_gen = any(m.name == configured_gen_model or m.name == f"models/{configured_gen_model}" for m in models)
        
        print(f"Configured Generation Model ({configured_gen_model}): {'FOUND' if found_gen else 'NOT FOUND'}")
        
        if not found_gen:
            print("Available generation models:")
            for m in models:
                if "generateContent" in m.supported_generation_methods:
                    print(f" - {m.name}")

        print("\nAttempting simple generation query...")
        test_model_name = configured_gen_model if found_gen else "gemini-1.5-flash"
        if not test_model_name.startswith("models/"):
             test_model_name = f"models/{test_model_name}"
             
        try:
            model = genai.GenerativeModel(test_model_name)
            response = model.generate_content("Reply with 'OK'")
            print(f"Generation response: {response.text}")
            print("API Key is VALID and WORKING.")
        except Exception as gen_error:
            print(f"Generation Failed with model {test_model_name}: {gen_error}")

    except Exception as api_error:
        print(f"API Call Failed: {api_error}")
        print("The API Key may be invalid, expired, or lacking permissions.")

except Exception as e:
    print(f"Script Error: {e}")
