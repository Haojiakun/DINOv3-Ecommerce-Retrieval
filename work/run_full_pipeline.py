"""Run the full pipeline on the balanced 5K subset.

Steps:
  1. extract_features.py    -> cls.npy, patch.npy, meta.json
  2. build_text.py          -> text_feats.npy
  3. extract_attrs_vqa.py   -> vqa_attrs.json  (slow, ~1.5h)
  4. filter_attrs.py        -> attr_emb_*.npy, attr_label_*.npy
  5. train_visual.py        -> visual_head.pt
  6. run_seeds_local.py     -> seeds_local_results.md

Usage:
    work\\venv\\Scripts\\python.exe work\\run_full_pipeline.py
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")
FEAT = os.path.join(WORK, "features")
PY = os.path.join(WORK, "venv", "Scripts", "python.exe")

STEPS = [
    ("extract_features.py", "DINOv3 feature extraction"),
    ("build_text.py", "CLIP text features"),
    ("extract_attrs_vqa.py", "BLIP-VQA attribute extraction (SLOW)"),
    ("filter_attrs.py", "attribute filtering + CLIP embeddings"),
    ("train_visual.py", "visual head pre-training (SupCon)"),
    ("run_seeds_local.py", "3-seed cross-modal alignment + eval"),
]


def backup_old():
    bak = FEAT + "_menswear_backup"
    if os.path.isdir(FEAT) and not os.path.isdir(bak):
        print(f"[backup] moving old features/ -> features_menswear_backup/")
        shutil.move(FEAT, bak)
    elif os.path.isdir(bak):
        print(f"[backup] features_menswear_backup/ already exists, skipping")
    os.makedirs(FEAT, exist_ok=True)


def run(script, label):
    path = os.path.join(WORK, script)
    print(f"\n{'='*60}")
    print(f"[STEP] {label}: {script}")
    print(f"{'='*60}")
    t0 = time.time()
    env = dict(os.environ)
    proc = subprocess.run([PY, path], cwd=WORK, env=env)
    dt = time.time() - t0
    print(f"[DONE] {script} in {dt/60:.1f} min (exit={proc.returncode})")
    if proc.returncode != 0:
        print(f"[FATAL] {script} failed, aborting.")
        sys.exit(1)


def wait_for_gpu(max_wait_hours=12):
    """Wait until GPU memory usage drops below threshold."""
    import subprocess
    import time
    deadline = time.time() + max_wait_hours * 3600
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True
            ).strip()
            mem_used = int(out.split("\n")[0].strip())
            if mem_used < 1500:
                print(f"[GPU] idle ({mem_used} MiB used), starting pipeline")
                return True
        except Exception:
            pass
        print(f"[GPU] busy ({mem_used if 'mem_used' in dir() else '?'} MiB), waiting 60s...", flush=True)
        time.sleep(60)
    print("[GPU] wait timeout, starting anyway")
    return False


def main():
    wait_for_gpu()
    backup_old()
    for script, label in STEPS:
        run(script, label)
    print("\n" + "=" * 60)
    print("[ALL DONE] Pipeline complete. Check outputs/seeds_local_results.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
