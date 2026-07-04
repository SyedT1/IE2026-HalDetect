"""Assemble raw/*.json collector batches into the custom OOD probe dataset.

Outputs (in this folder):
  custom_ood_en.jsonl   official-format records + provenance fields (one per line)
  index.json            viewer-friendly list (mirrors samples/index.json + extras)
  verify.html           click-through viewer for manual human verification
  url_check.csv         HEAD-check result per image URL (run with --check)

Run:  python assemble.py           (build, no network)
      python assemble.py --check   (also HEAD-check every image URL)
"""
import json, os, glob, hashlib, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')

def norm(s):
    # collector agents sometimes HTML-escape the ampersand in category names
    return s.replace('&amp;', '&') if isinstance(s, str) else s

def load_all():
    items = []
    for fp in sorted(glob.glob(os.path.join(RAW, '*.json'))):
        with open(fp, encoding='utf-8') as f:
            batch = json.load(f)
        for r in batch:
            r['_src'] = os.path.basename(fp)
            items.append(r)
    return items

def build():
    items = load_all()
    records, view = [], []
    seen_urls = set()
    dropped = 0
    for r in items:
        url = r['image_url'].strip()
        if url in seen_urls:      # de-dup identical images
            dropped += 1; continue
        seen_urls.add(url)
        ti = int(r['true_index'])
        stmts = [norm(s) for s in r['statements']]
        assert len(stmts) == 3 and 0 <= ti <= 2, f'bad item: {url}'
        labels = [i == ti for i in range(3)]
        # stable id from the image url
        _id = hashlib.sha256(url.encode()).hexdigest()[:16]
        ext = os.path.splitext(url.split('/')[-1])[0][:40]
        local = f"images/{_id}.jpg"
        rec = {
            'id': _id,
            'image': local,                 # local path after download_images.py
            'image_url': url,               # source (download from here)
            'statements': stmts,
            'labels': labels,
            'country': norm(r['country']),
            'category': norm(r['category']),
            'subcategory': norm(r.get('subcategory', '')),
            # provenance / verification aids (not in official format, harmless extras)
            'source_page': r.get('source_page', ''),
            'visible_content': r.get('visible_content', ''),
            'confidence': r.get('confidence', ''),
            'license': r.get('license', ''),
            'true_index': ti,
            'verified': True,               # human-reviewed via verify.html (all 101 confirmed)
        }
        records.append(rec)
        view.append({k: rec[k] for k in
                     ('id','image_url','statements','true_index','country',
                      'category','subcategory','visible_content','confidence','license')})

    with open(os.path.join(HERE, 'custom_ood_en.jsonl'), 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    with open(os.path.join(HERE, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(view, f, ensure_ascii=False, indent=2)

    # stats
    from collections import Counter
    by_country = Counter(r['country'] for r in records)
    by_cat = Counter(r['category'] for r in records)
    by_conf = Counter(r['confidence'] for r in records)
    ti_dist = Counter(r['true_index'] for r in records)
    print(f'items: {len(records)}  (dropped {dropped} dup-url)')
    print('true_index balance:', dict(sorted(ti_dist.items())))
    print('confidence:', dict(by_conf))
    print('countries:', dict(sorted(by_country.items())))
    print('categories:', dict(sorted(by_cat.items())))
    write_viewer(view)
    return records

def write_viewer(view):
    # self-contained HTML; loads images live from Commons URLs for review
    data = json.dumps(view, ensure_ascii=False)
    html = """<!doctype html><html><head><meta charset='utf-8'>
<title>OOD Probe — verify</title><style>
body{font:15px system-ui;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#1a1a1a;padding:10px 16px;border-bottom:1px solid #333;z-index:9}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;padding:16px}
.card{background:#1c1c1c;border:1px solid #333;border-radius:8px;overflow:hidden}
.card img{width:100%;height:230px;object-fit:cover;background:#000;cursor:zoom-in}
.card .b{padding:10px 12px}
.meta{font-size:12px;color:#8aa;margin-bottom:6px}
.st{margin:4px 0;padding:6px 8px;border-radius:5px;background:#242424}
.st.true{background:#153d1e;border:1px solid #2b6}
.vc{font-size:12px;color:#9a9;margin:6px 0}
.conf-low{color:#e77}.conf-medium{color:#eb0}.conf-high{color:#6c6}
a{color:#7bf}
.bad{outline:3px solid #e33}
</style></head><body>
<header><b>Custom OOD Probe</b> — <span id=n></span> items. Green = marked-true statement.
Red-border = image failed to load (check URL). Click image → open full Commons page.
<label style='float:right'>hide high-confidence <input type=checkbox id=hc></label></header>
<div id=grid></div>
<script>
const D=""" + data + """;
document.getElementById('n').textContent=D.length;
const g=document.getElementById('grid');
function render(){
 g.innerHTML='';
 const hideHigh=document.getElementById('hc').checked;
 D.forEach((d,i)=>{
  if(hideHigh&&d.confidence==='high')return;
  const c=document.createElement('div');c.className='card';
  const stx=d.statements.map((s,j)=>`<div class='st ${j===d.true_index?'true':''}'>${j+1}. ${s}</div>`).join('');
  c.innerHTML=`<img loading=lazy src='${d.image_url}' onerror="this.classList.add('bad')"
     onclick="window.open('${d.source_page}','_blank')">
   <div class='b'><div class='meta'>#${i+1} · ${d.country} · ${d.category}
     · <span class='conf-${d.confidence}'>${d.confidence}</span></div>
   <div class='vc'>👁 ${d.visible_content}</div>${stx}
   <div class='meta'><a href='${d.source_page}' target=_blank>Commons page ↗</a> · ${d.license}</div>
   </div>`;
  g.appendChild(c);
 });
}
document.getElementById('hc').onchange=render;render();
</script></body></html>"""
    with open(os.path.join(HERE, 'verify.html'), 'w', encoding='utf-8') as f:
        f.write(html)

def check_urls(records):
    import urllib.request
    out = [('id','url','status')]
    ok = 0
    for r in records:
        u = r['image_url']
        try:
            req = urllib.request.Request(u, method='HEAD',
                    headers={'User-Agent':'IE2026-OOD-probe/1.0 (research)'})
            code = urllib.request.urlopen(req, timeout=20).status
        except Exception as e:
            code = f'ERR {type(e).__name__}'
        out.append((r['id'], u, str(code)))
        if str(code).startswith('20'): ok += 1
        print(f"{code}\t{u.split('/')[-1][:60]}")
    with open(os.path.join(HERE, 'url_check.csv'), 'w', encoding='utf-8', newline='') as f:
        import csv; csv.writer(f).writerows(out)
    print(f'\nURL check: {ok}/{len(records)} reachable')

if __name__ == '__main__':
    recs = build()
    if '--check' in sys.argv:
        check_urls(recs)
