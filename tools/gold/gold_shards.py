#!/usr/bin/env python3
"""Shard finalized synthetic gold pages and upload to a PRIVATE Hugging Face repo.

Durable/resumable. Runs forever:
  - packs unsharded PASS pages (PNG + GT + meta) into ~1000-page tar.gz shards (1-5 GB cap)
  - records SHA256 per shard
  - uploads to HF dataset repo (private) using HF_TOKEN from /home/jovyan/.secrets/hf_token
  - verifies by re-downloading and comparing SHA256
  - updates progress.json
"""
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

GOLD = Path("/home/jovyan/gold")
STATE = GOLD / "state"
FINAL = GOLD / "synthetic_gold"
SHARDS = GOLD / "shards"
PROG = STATE / "progress.json"
HF_REPO = "cloudaocr/gold-15k-synthetic-private"
MAX_PAGES_PER_SHARD = 1000
MAX_SHARD_BYTES = 5_000_000_000


def log(m):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}", flush=True)


def load_state():
    return json.loads(PROG.read_text()) if PROG.exists() else {"synthetic": {}, "shards": {"finalized": [], "uploaded": []}}


def save_state(st):
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    PROG.write_text(json.dumps(st, indent=1, ensure_ascii=False))


def hf_token():
    return open("/home/jovyan/.secrets/hf_token").read().strip()


def build_shard(shard_idx, page_ids):
    shard_path = SHARDS / f"shard_{shard_idx:04d}.tar.gz"
    if shard_path.exists():
        return shard_path
    with tarfile.open(shard_path, "w:gz") as tf:
        count = 0
        for pid in page_ids:
            for suffix in (".png", ".gt.txt", ".meta.json"):
                f = FINAL / f"{pid}{suffix}"
                if f.exists():
                    tf.add(f, arcname=f"{pid}{suffix}")
            count += 1
            if count >= MAX_PAGES_PER_SHARD:
                break
    return shard_path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    SHARDS.mkdir(parents=True, exist_ok=True)
    os.environ["HF_TOKEN"] = hf_token()
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    try:
        api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
        log(f"HF repo ready: {HF_REPO} (private)")
    except Exception as e:
        log(f"HF repo create failed: {e}")

    while True:
        st = load_state()
        try:
            all_ids = sorted({p.name.split(".")[0] for p in FINAL.glob("CLD-SG-*.png")})
            done_ids = set()
            for s in st.get("shards", {}).get("finalized", []):
                done_ids.update(s["page_range"])
            todo = [i for i in all_ids if i not in done_ids]
            if len(todo) >= 1000 or (todo and len(all_ids) >= 7167):
                batch = todo[:MAX_PAGES_PER_SHARD]
                idx = len(st.get("shards", {}).get("finalized", [])) + 1
                sp = build_shard(idx, batch)
                digest = sha256_file(sp)
                size = sp.stat().st_size
                entry = {"shard": sp.name, "sha256": digest, "bytes": size,
                         "page_range": batch[:MAX_PAGES_PER_SHARD]}
                # upload
                try:
                    api.upload_file(path_or_fileobj=str(sp), path_in_repo=sp.name,
                                    repo_id=HF_REPO, repo_type="dataset")
                    # verify by re-download
                    dl = SHARDS / f"verify_{sp.name}"
                    import urllib.request
                    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{sp.name}"
                    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {hf_token()}"})
                    with urllib.request.urlopen(req, timeout=600) as r, open(dl, "wb") as f:
                        f.write(r.read())
                    ok = sha256_file(dl) == digest
                    dl.unlink()
                    entry["verified"] = ok
                    log(f"shard {sp.name}: {size} bytes uploaded verified={ok}")
                except Exception as e:
                    entry["verified"] = False
                    entry["upload_error"] = str(e)[:200]
                    log(f"shard {sp.name}: upload FAILED {e}")
                st.setdefault("shards", {"finalized": [], "uploaded": []}).setdefault("finalized", []).append(entry)
                if entry.get("verified"):
                    st["shards"].setdefault("uploaded", []).append(sp.name)
                save_state(st)
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}")
        time.sleep(120)


if __name__ == "__main__":
    main()
