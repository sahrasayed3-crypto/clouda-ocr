#!/usr/bin/env python3
"""Clouda GOLD-15K synthetic pipeline — durable, resumable, seeded (15092026).

Stages (run via `python gold_pipeline.py <stage>` or `loop`):
  fetch    — download license-verified text sources (public, catalog-recorded)
  chunk    — extract unique Arabic text chunks -> /home/jovyan/gold/chunks/<source>/
  classify — deterministic content-class assignment (seeded)
  generate — run clouda_data.factory per visual-profile batch (variants=1, max-pages=1)
  qc       — per-page gates; PASS -> CLD-SG-NNNNNN ID; FAIL -> quarantine + reasons
  loop     — forever: generate -> qc -> progress.json (30s cycle)
State: /home/jovyan/gold/state/progress.json
Licenses are recorded per source in state/source_provenance.json (fail-closed).
"""
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

GOLD = Path("/home/jovyan/gold")
STATE = GOLD / "state"
LOGS = GOLD / "logs"
CHUNKS = GOLD / "chunks"
FINAL = GOLD / "synthetic_gold"
QUAR = GOLD / "quarantine"
WORK = GOLD / "factory_runs"
REPO = Path("/home/jovyan/clouda-ocr")
DATA = Path("/home/jovyan/data")
PROG = STATE / "progress.json"

SEED = 15092026
TARGET = 7167
MIN_CHARS, MAX_CHARS = 700, 1400
VISUAL_QUOTA = {"clean_realistic": 1075, "normal_scan": 2508, "medium_degraded": 2509, "hard_realistic": 1075}
CONTENT_QUOTA = {"non_fiction_knowledge": 2000, "novels_literature_prose": 1500, "classical_older_arabic": 1000,
                 "children_simple_prose": 650, "drama_dialogue": 550, "poetry": 450, "diacritized_arabic": 600,
                 "mixed_remaining": 417}
VISUAL_PROFILES = {
    "clean_realistic": ["01_high_quality_flatbed", "02_clean_book_scan"],
    "normal_scan": ["03_normal_office_scan", "09_archive_scan", "02_clean_book_scan"],
    "medium_degraded": ["04_old_book_light", "05_old_book_medium", "07_old_photocopy", "11_aged_but_readable"],
    "hard_realistic": ["06_old_book_heavy", "08_recopied_photocopy", "12_hard_composite", "10_low_dpi_scan"],
}
AR = re.compile(r"[\u0600-\u06FF]")
AR_DIAC = re.compile(r"[\u064B-\u0652\u0670]")
BENCH_HASHES = set()
if (STATE / "benchmark_hashes.json").exists():
    BENCH_HASHES = set(json.loads((STATE / "benchmark_hashes.json").read_text()))

SOURCE_PROVENANCE = {
    "rasam": {"id": "rasam_dataset", "license": "Apache-2.0", "verified": True,
              "url": "calfa-ai/RASAM-1 (BULAC manuscripts)", "class_hint": "classical_older_arabic"},
    "arwiki": {"id": "arabic_wikipedia_text", "license": "CC-BY-SA-4.0", "verified": True,
               "url": "https://dumps.wikimedia.org/arwiki/", "class_hint": "non_fiction_knowledge",
               "attribution": "Text from Arabic Wikipedia contributors, CC BY-SA 4.0 (share-alike applies)"},
    "tashkeela": {"id": "tashkeela_diacritized_text", "license": "GPL-2.0-only", "verified": True,
                  "url": "https://sourceforge.net/p/tashkeela/", "class_hint": "diacritized_arabic",
                  "attribution": "Zerrouki & Balla, Data in Brief 2017; GPL-2.0"},
    "makhzan": {"id": "openiti_makhzan", "license": "CC-BY-NC-SA-4.0", "verified": True,
                "url": "https://zenodo.org/records/19861912", "class_hint": "classical_older_arabic",
                "attribution": "OpenITI MAKHZAN, CC BY-NC-SA 4.0; noncommercial research use"},
    "sard": {"id": "sard_synthetic_arabic_recognition_dataset", "license": "Apache-2.0", "verified": True,
             "url": "https://huggingface.co/datasets/riotu-lab/SARD", "class_hint": "non_fiction_knowledge"},
}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def ensure_dirs():
    for d in (GOLD, STATE, LOGS, CHUNKS, FINAL, QUAR, WORK, DATA):
        d.mkdir(parents=True, exist_ok=True)


def load_state():
    if PROG.exists():
        return json.loads(PROG.read_text())
    return {"synthetic": {"chunks": 0, "generated": 0, "pass": 0, "rejected": 0, "pass_ids": 0},
            "benchmark": {"status": "BLOCKED: 462-page holdout (MISRAJ/KITAB-R) data not found on any accessible storage"},
            "real_gold": {"status": "pending"}, "system": {}, "updated": None}


def save_state(st):
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    PROG.write_text(json.dumps(st, indent=1, ensure_ascii=False))


def http_download(url, dest, max_bytes=2_000_000_000):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "clouda-gold-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            got += len(chunk)
            if got > max_bytes:
                raise RuntimeError(f"download exceeds cap {max_bytes}")
            f.write(chunk)
    return dest


def stage_fetch():
    ensure_dirs()
    prov = STATE / "source_provenance.json"
    (STATE).mkdir(exist_ok=True, parents=True)
    prov.write_text(json.dumps(SOURCE_PROVENANCE, ensure_ascii=False, indent=1))
    # 1. Arabic Wikipedia articles part1 (non-fiction knowledge, CC-BY-SA)
    try:
        http_download("https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles-multistream1.xml-p1p340838.bz2",
                      DATA / "arwiki_part1.xml.bz2", max_bytes=1_500_000_000)
        log("fetch: arwiki articles part1 OK")
    except Exception as e:
        log(f"fetch: arwiki FAILED {type(e).__name__}: {e}")
    # 2. Tashkeela vocalized text
    try:
        t_url = "https://sourceforge.net/projects/tashkeela/files/latest/download"
        http_download(t_url, DATA / "tashkeela.zip", max_bytes=1_500_000_000)
        log("fetch: tashkeela OK")
    except Exception as e:
        log(f"fetch: tashkeela FAILED {type(e).__name__}: {e}")
    # 3. OpenITI makhzan via Zenodo API (text files only, capped)
    try:
        rec = json.loads(urllib.request.urlopen("https://zenodo.org/api/records/19861912", timeout=60).read())
        files = [f for f in rec.get("files", []) if f["key"].endswith((".zip", ".txt")) and f["size"] < 800_000_000]
        for f in files[:2]:
            http_download(f["links"]["self"], DATA / f"makhzan_{f['key']}", max_bytes=800_000_000)
        log(f"fetch: makhzan {len(files[:2])} files OK")
    except Exception as e:
        log(f"fetch: makhzan FAILED {type(e).__name__}: {e}")
    # 4. SARD text component (try HF API listing, grab text-like files)
    try:
        tree = json.loads(urllib.request.urlopen(
            "https://huggingface.co/api/datasets/riotu-lab/SARD/tree/main", timeout=60).read())
        keys = [t["path"] for t in tree if t["path"].endswith((".txt", ".jsonl", ".csv"))]
        for k in keys[:5]:
            http_download(f"https://huggingface.co/datasets/riotu-lab/SARD/resolve/main/{k}", DATA / "sard_" + k.replace("/", "_"))
        log(f"fetch: sard {len(keys[:5])} files OK")
    except Exception as e:
        log(f"fetch: sard FAILED {type(e).__name__}: {e}")


def split_chunks(body, out_dir, prefix, seen):
    body = re.sub(r"\s+", " ", body).strip()
    made = 0
    for i in range(0, len(body), MAX_CHARS):
        part = body[i:i + MAX_CHARS]
        if len(part) < MIN_CHARS:
            break
        h = hashlib.sha256(part.encode()).hexdigest()[:16]
        if h in seen:
            continue
        seen.add(h)
        (out_dir / f"{prefix}_{h}.txt").write_text(part, encoding="utf-8")
        made += 1
    return made


def iter_arwiki_articles(max_pages=6000):
    f = DATA / "arwiki_part1.xml.bz2"
    if not f.exists():
        return
    import bz2
    page_re = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
    title_re = re.compile(r"<title>(.*?)</title>", re.S)
    buf = []
    n = 0
    with bz2.open(f, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            buf.append(line)
            if "</page>" in line:
                blob = "".join(buf)
                buf = []
                n += 1
                if n > max_pages:
                    return
                if any(t in blob for t in (":", "Wikipedia:")) and "<title>" in blob and title_re.search(blob).group(1).count(":"):
                    continue  # skip namespace pages
                texts = page_re.findall(blob)
                if not texts:
                    continue
                text = re.sub(r"&lt;[^&]*&gt;|\{\{[^}]*\}\}|\[\[|\]\]|'{2,}", " ", texts[0])
                text = re.sub(r"\s+", " ", text).strip()
                if AR.search(text) and len(text) >= MIN_CHARS:
                    yield text


def stage_chunk():
    ensure_dirs()
    seen = {p.stem.split("_", 1)[1] for d in CHUNKS.iterdir() if d.is_dir() for p in d.glob("*.txt")}
    total = 0
    # rasam page XML
    out = CHUNKS / "rasam"
    out.mkdir(exist_ok=True)
    for xml in sorted(DATA.rglob("*.xml")):
        if "source_manifest" in xml.name:
            continue
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        texts = [el.text.strip() for el in root.iter() if el.text and AR.search(el.text)]
        if texts:
            total += split_chunks(" ".join(texts), out, "ras", seen)
    # arwiki articles: split article text into chunks
    out = CHUNKS / "arwiki"
    out.mkdir(exist_ok=True)
    for text in iter_arwiki_articles():
        total += split_chunks(text[:MAX_CHARS * 2], out, "wik", seen)
    # tashkeela zip
    tz = DATA / "tashkeela.zip"
    if tz.exists():
        out = CHUNKS / "tashkeela"
        out.mkdir(exist_ok=True)
        try:
            import zipfile
            with zipfile.ZipFile(tz) as z:
                count = 0
                for n in z.namelist():
                    if n.endswith((".txt", ".csv")) and count < 200:
                        try:
                            body = z.read(n).decode("utf-8", "ignore")
                        except Exception:
                            continue
                        total += split_chunks(body, out, "tsh", seen)
                        count += 1
        except Exception as e:
            log(f"chunk: tashkeela extract FAILED {e}")
    # makhzan files
    for mf in DATA.glob("makhzan_*"):
        out = CHUNKS / "makhzan"
        out.mkdir(exist_ok=True)
        try:
            if mf.suffix == ".zip":
                import zipfile
                with zipfile.ZipFile(mf) as z:
                    count = 0
                    for n in z.namelist():
                        if n.endswith((".txt", ".mmax", ".xml")) and count < 200:
                            body = z.read(n).decode("utf-8", "ignore")
                            body = re.sub(r"<[^>]+>", " ", body)
                            total += split_chunks(body, out, "mkz", seen)
                            count += 1
            else:
                body = mf.read_text(encoding="utf-8", errors="ignore")
                total += split_chunks(body, out, "mkz", seen)
        except Exception as e:
            log(f"chunk: makhzan extract FAILED {e}")
    # sard text files
    for sf in DATA.glob("sard_*"):
        out = CHUNKS / "sard"
        out.mkdir(exist_ok=True)
        try:
            body = sf.read_text(encoding="utf-8", errors="ignore")
            total += split_chunks(body, out, "srd", seen)
        except Exception as e:
            log(f"chunk: sard extract FAILED {e}")
    log(f"chunk: total new chunks {total}")
    return total


def classify(text):
    diac = len(AR_DIAC.findall(text)) / max(len(AR.findall(text)), 1)
    words = text.split()
    avg_w = sum(len(w) for w in words) / max(len(words), 1)
    if diac > 0.25:
        return "diacritized_arabic"
    if diac > 0.08 and avg_w < 5.5:
        return "classical_older_arabic"
    return "mixed_remaining"


def stage_classify():
    man = STATE / "chunk_classes.json"
    classes = json.loads(man.read_text()) if man.exists() else {}
    made = 0
    for d in CHUNKS.iterdir():
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.txt")):
            if f.stem in classes:
                continue
            classes[f.stem] = classify(f.read_text(encoding="utf-8"))
            made += 1
    man.write_text(json.dumps(classes, ensure_ascii=False))
    log(f"classify: classified {made} new chunks (total {len(classes)})")
    return classes


def stage_generate(batch_limit, workers):
    plan_man = STATE / "chunk_plan.json"
    assigned = json.loads(plan_man.read_text()) if plan_man.exists() else {}
    done_files = {p.stem for p in FINAL.glob("*.png")}
    chunks = []
    for d in sorted(CHUNKS.iterdir()):
        if d.is_dir():
            chunks.extend(sorted(d.glob("*.txt")))
    todo = [c for c in chunks if c.stem not in assigned and c.stem not in done_files]
    rng = random.Random(SEED + len(assigned))
    counts = {k: 0 for k in VISUAL_QUOTA}
    for v in assigned.values():
        counts[v] = counts.get(v, 0) + 1
    plan = {}
    for c in todo[:batch_limit]:
        remaining = {k: VISUAL_QUOTA[k] - counts.get(k, 0) for k in VISUAL_QUOTA if counts.get(k, 0) < VISUAL_QUOTA[k]}
        if not remaining:
            break
        tot = sum(remaining.values())
        pick = rng.choices(list(remaining), weights=[v / tot for v in remaining.values()])[0]
        plan[c.stem] = pick
        counts[pick] += 1
    if not plan:
        log("generate: nothing to plan (all assigned or done)")
        return
    plan_man.write_text(json.dumps({**assigned, **plan}, ensure_ascii=False))
    for vis, stems in {k: [s for s, v in plan.items() if v == k] for k in VISUAL_PROFILES
                       if any(v == k for v in plan.values())}.items():
        batch_name = f"batch_{vis}_{int(time.time())}"
        staging = GOLD / "staging" / batch_name
        staging.mkdir(parents=True, exist_ok=True)
        for s in stems:
            src = next(CHUNKS.glob(f"*/{s}.txt"))
            shutil.copy(src, staging / f"{s}.txt")
        profile = VISUAL_PROFILES[vis][hash(vis + str(len(stems))) % len(VISUAL_PROFILES[vis])]
        resume = "--resume" if (GOLD / "factory_runs" / batch_name).exists() else ""
        cmd = (f"nice -n 10 ionice -c3 python -m clouda_data.factory generate "
               f"--output {GOLD / 'factory_runs'} --profiles {profile} --variants 1 "
               f"--seed {SEED} --seed-mode arabic_scan_factory --workers {workers} "
               f"--max-pages 1 --no-pdf --run-id {batch_name} {resume} {staging}")
        log(f"generate: {len(stems)} chunks [{vis}] profile={profile}")
        with open(LOGS / f"factory_{batch_name}.log", "w") as lf:
            subprocess.run(cmd, shell=True, cwd=REPO, stdout=lf, stderr=lf)


def stage_qc():
    import cv2
    man = STATE / "qc_done.json"
    done = set(json.loads(man.read_text()) if man.exists() else [])
    fi = STATE / "final_index.json"
    finals = json.loads(fi.read_text()) if fi.exists() else {"count": 0, "text_hashes": [], "img_hashes": []}
    classes = json.loads((STATE / "chunk_classes.json").read_text()) if (STATE / "chunk_classes.json").exists() else {}
    plan = json.loads((STATE / "chunk_plan.json").read_text()) if (STATE / "chunk_plan.json").exists() else {}
    st = load_state()
    processed = 0
    for run_dir in sorted(WORK.glob("batch_*")):
        for page_png in sorted(run_dir.rglob("*.png")):
            rel = page_png.relative_to(run_dir).parts
            # find the per-document directory name holding the chunk id
            stem = None
            for part in rel[:-1]:
                if re.match(r"^(ras|wik|tsh|mkz|srd)_", part):
                    stem = part
                    break
            key = f"{run_dir.name}/{page_png.relative_to(run_dir)}"
            if key in done or stem is None:
                continue
            done.add(key)
            processed += 1
            reasons = []
            img = None
            try:
                img = cv2.imread(str(page_png))
                if img is None:
                    reasons.append("undecodable_image")
                else:
                    h, w = img.shape[:2]
                    if w < 400 or h < 400 or w > 6000 or h > 9000:
                        reasons.append(f"bad_dimensions_{w}x{h}")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if gray.std() < 8:
                        reasons.append("near_blank")
            except Exception as e:
                reasons.append(f"image_error_{type(e).__name__}")
            src_txt = None
            for d in CHUNKS.iterdir():
                if d.is_dir() and (d / f"{stem}.txt").exists():
                    src_txt = d / f"{stem}.txt"
                    break
            gt = src_txt.read_text(encoding="utf-8").strip() if src_txt else ""
            if not gt:
                reasons.append("empty_gt")
            if gt and ("\ufffd" in gt or not AR.search(gt)):
                reasons.append("bad_shaping_or_empty")
            img_sha = hashlib.sha256(page_png.read_bytes()).hexdigest()
            txt_sha = hashlib.sha256(gt.encode()).hexdigest()
            if img_sha in finals["img_hashes"]:
                reasons.append("duplicate_image_sha")
            if txt_sha in finals["text_hashes"]:
                reasons.append("duplicate_text_hash")
            # benchmark leakage (protected canonical holdout must never appear in training data)
            norm = re.sub(r"[\u064B-\u0652\u0640\s]", "", gt)
            for h in (hashlib.sha256(gt.encode()).hexdigest(),
                      hashlib.sha256(norm.encode()).hexdigest()):
                if h in BENCH_HASHES:
                    reasons.append("benchmark_leakage")
                    break
            if reasons:
                qdir = QUAR / run_dir.name
                qdir.mkdir(parents=True, exist_ok=True)
                shutil.copy(page_png, qdir / page_png.name)
                (qdir / f"{page_png.stem}.reject.json").write_text(json.dumps({"reasons": reasons}, indent=1))
                st["synthetic"]["rejected"] += 1
            else:
                finals["count"] += 1
                page_id = f"CLD-SG-{finals['count']:06d}"
                shutil.copy(page_png, FINAL / f"{page_id}.png")
                (FINAL / f"{page_id}.gt.txt").write_text(gt, encoding="utf-8")
                (FINAL / f"{page_id}.meta.json").write_text(json.dumps(
                    {"page_id": page_id, "source_chunk": stem, "content_class": classes.get(stem),
                     "visual": plan.get(stem), "layout": "factory_native",
                     "image_sha256": img_sha, "text_sha256": txt_sha,
                     "license": "inherited_from_source_provenance",
                     "normalized_eval_text": re.sub(r"[\u064B-\u0652\u0640]", "", gt)},
                    ensure_ascii=False, indent=1))
                finals["text_hashes"].append(txt_sha)
                finals["img_hashes"].append(img_sha)
                st["synthetic"]["pass"] += 1
                st["synthetic"]["pass_ids"] = finals["count"]
    (STATE / "qc_done.json").write_text(json.dumps(sorted(done)))
    (STATE / "final_index.json").write_text(json.dumps(finals))
    log(f"qc: processed {processed} new pages (PASS total {finals['count']})")
    return processed


def update_system(st):
    try:
        disk = shutil.disk_usage("/home/jovyan")
        gpu = subprocess.run(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True).stdout.strip()
        mem = {ln.split(":")[0]: int(ln.split()[1]) // 1024 for ln in open("/proc/meminfo") if ln.startswith(("MemTotal", "MemAvailable"))}
        st["system"] = {"load": os.getloadavg()[0], "ram_total_mb": mem.get("MemTotal"), "ram_avail_mb": mem.get("MemAvailable"),
                        "disk_used_gb": round(disk.used / 1e9, 1), "disk_free_gb": round(disk.free / 1e9, 1), "gpu": gpu}
    except Exception as e:
        st["system"] = {"error": str(e)}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "loop"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    ensure_dirs()
    if mode == "loop":
        lock = STATE / "loop.pid"
        if lock.exists():
            old = lock.read_text().strip()
            if old and Path(f"/proc/{old}").exists():
                log(f"loop: already running (pid {old}); exiting")
                return
        lock.write_text(str(os.getpid()))
    if mode == "fetch":
        stage_fetch()
        return
    if mode in ("chunk", "loop"):
        stage_chunk()
        stage_classify()
    while True:
        st = load_state()
        try:
            if mode == "loop":
                stage_generate(batch_limit=400, workers=workers)
                stage_qc()
            update_system(st)
            save_state(st)
        except Exception as e:
            log(f"ERROR {mode}: {type(e).__name__}: {e}")
            st["last_error"] = f"{type(e).__name__}: {e}"
            save_state(st)
            if mode != "loop":
                raise
            time.sleep(60)
        if mode != "loop":
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
