import os
import requests
from pathlib import Path
import time

# 1. Setup paths - Go to the PROJECT ROOT
_ROOT = Path(__file__).resolve().parent.parent.parent
final_dir = _ROOT / "data" / "stl10_test_input"
final_dir.mkdir(parents=True, exist_ok=True)

# 2. Keywords for variety
search_terms = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

print(f"🚀 Simulating real user input into: {final_dir}")

# We'll use a more stable "Lorem Flickr" service which is great for testing
for term in search_terms:
    print(f"📸 Fetching 20 real-world {term} photos...")
    for i in range(20):
        try:
            # LoremFlickr is very stable for automated testing
            url = f"https://loremflickr.com/800/600/{term}"
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                # Save with a name that looks like a user's phone upload
                filename = f"IMG_{time.strftime('%Y%m%d')}_{term}_{i:03d}.jpg"
                with open(final_dir / filename, 'wb') as f:
                    f.write(response.content)
                print(f"  ✅ Saved {filename}")
            
            # Tiny sleep to prevent being flagged
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  ⚠️ Error on {term}: {e}")

print(f"\n✅ DONE. Check your folder: {final_dir}")