"""Download the OOD-probe images from their Commons URLs into ./images/.
Run AFTER human verification, once, to make the set self-contained (Kaggle-uploadable).
Skips files already present; prints any that fail so you can fix or drop them.

  python download_images.py
"""
import json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, 'images'); os.makedirs(IMG, exist_ok=True)
UA = {'User-Agent': 'IE2026-OOD-probe/1.0 (research; contact via repo)'}

rows = [json.loads(l) for l in open(os.path.join(HERE, 'custom_ood_en.jsonl'), encoding='utf-8')]
ok = fail = skip = 0
for r in rows:
    dst = os.path.join(HERE, r['image'])          # images/<id>.jpg
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        skip += 1; continue
    try:
        req = urllib.request.Request(r['image_url'], headers=UA)
        data = urllib.request.urlopen(req, timeout=40).read()
        with open(dst, 'wb') as f:
            f.write(data)
        ok += 1
        print(f"OK  {r['country']:14} {r['id']}  ({len(data)//1024} KB)")
        time.sleep(0.3)                            # be polite to Commons
    except Exception as e:
        fail += 1
        print(f"FAIL {r['id']}  {type(e).__name__}: {r['image_url']}")
print(f"\ndownloaded {ok}, skipped {skip}, failed {fail}  ->  {IMG}")
if fail:
    print("Re-run to retry failures, or drop those rows from custom_ood_en.jsonl.")
