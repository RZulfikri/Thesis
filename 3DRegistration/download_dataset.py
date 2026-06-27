import requests
import sys

url = "https://nghiaho.com/uploads/box_can.zip"
print(f"Downloading {url}...")
try:
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        dl = 0
        with open("box_can.zip", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                dl += len(chunk)
                done = int(50 * dl / total_size)
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl}/{total_size}")
                sys.stdout.flush()
    print("\nDownload complete.")
except Exception as e:
    print(f"\nDownload failed: {e}")
