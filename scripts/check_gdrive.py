import urllib.request
import re

url = "https://drive.google.com/drive/folders/1Siy_98g0FZ4XNJSUb2XoaIeW2_dwYqVS?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
        print("Page fetched, length:", len(html))
        # Look for Bijie or Landslide4Sense
        for m in re.finditer(r'([a-zA-Z0-9_\-\.]+Bijie[a-zA-Z0-9_\-\.]*)', html, re.IGNORECASE):
            print("Found match:", m.group(0))
        for m in re.finditer(r'([a-zA-Z0-9_\-\.]+Landslide[a-zA-Z0-9_\-\.]*)', html, re.IGNORECASE):
            print("Found match:", m.group(0))
        # Look for file IDs
        file_ids = re.findall(r'\["([a-zA-Z0-9_-]{28,})"', html)
        print("Possible file IDs count:", len(set(file_ids)))
        for fid in list(set(file_ids))[:10]:
            print("  Candidate ID:", fid)
except Exception as e:
    print("Error:", e)
