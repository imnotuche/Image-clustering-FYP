import os
import requests
from pathlib import Path
import time

#Setup paths
_ROOT = Path(__file__).resolve().parent.parent.parent
final_dir = _ROOT / "data" / "stl10_test_input"
final_dir.mkdir(parents=True, exist_ok=True)

#Keywords for variety
search_terms = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

print(f"Simulating real user input into: {final_dir}")

#download 20 images per search term
for term in search_terms:
    print(f"Fetching 20 real-world {term} photos...")
    for i in range(20):
        try:
            
            url = f"https://loremflickr.com/800/600/{term}"
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                #save with a name that looks like a user's phone upload
                filename = f"IMG_{time.strftime('%Y%m%d')}_{term}_{i:03d}.jpg"
                with open(final_dir / filename, 'wb') as f:
                    f.write(response.content)
                print(f"  ✅ Saved {filename}")
            
            # tiny sleep to prevent being flagged
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error on {term}: {e}")

print(f"\nDONE. Check your folder: {final_dir}")