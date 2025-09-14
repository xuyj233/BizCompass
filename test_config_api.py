#!/usr/bin/env python3
"""
Simple config and API test
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_config():
    """Test config loading"""
    print("Testing config loading...")
    
    try:
        from config import create_default_config
        print("✅ Config import successful")
        
        config = create_default_config(debug_evaluation=False)
        print("✅ Config creation successful")
        
        print(f"API Key: {'Set' if config.api.api_key else 'Not set'}")
        print(f"Base URL: {config.api.base_url}")
        print(f"Model: {config.model.model_name}")
        
        return config
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return None

def test_api_call(config):
    """Test API call using requests with streaming"""
    if not config:
        return
        
    print("\nTesting API call with requests...")
    
    try:
        import requests
        import json
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {config.api.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare data
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        data = {
            'model': config.model.model_name,
            'messages': messages,
            'stream': True,
            'temperature': 0.0,
            'max_tokens': 100
        }
        
        print(f"📤 Sending request to: {config.api.base_url}/v1/chat/completions")
        print(f"   Model: {config.model.model_name}")
        
        # Make streaming request
        response = requests.post(
            f"{config.api.base_url}/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            stream=True
        )
        
        print(f"✅ API call successful (status: {response.status_code})")
        
        # Process streaming response
        full_content = ""
        print("📥 Streaming response:")
        
        for chunk in response.iter_lines():
            if chunk:
                decoded_chunk = chunk.decode('utf-8')
                if decoded_chunk.startswith('data:'):
                    try:
                        # Remove the 'data: ' prefix and parse the JSON object
                        parsed_chunk = json.loads(decoded_chunk[5:])
                        
                        if 'choices' in parsed_chunk and parsed_chunk['choices']:
                            delta = parsed_chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                print(content, end='')
                                full_content += content
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        print(f"\n❌ Error parsing chunk: {e}")
        
        print(f"\n\n📝 Full response content:")
        print(f"   Content: {full_content}")
        
        # Try to parse as JSON
        try:
            parsed_content = json.loads(full_content)
            print(f"   Parsed JSON: {parsed_content}")
        except json.JSONDecodeError:
            print(f"   Not valid JSON format")
            
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print(f"   Error type: {type(e)}")

if __name__ == "__main__":
    config = test_config()
    test_api_call(config)
