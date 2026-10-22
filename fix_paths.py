"""Fix hardcoded paths in cloud_run scripts to use relative paths."""
import os
import re

CLOUD = r"d:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper\cloud_run\work"

OLD_ROOT = r'D:\ASUS\Documents\Codex\2026-10-18\idea-x7167-0-1-x7684-paper'
NEW_ROOT = 'os.path.dirname(os.path.dirname(os.path.abspath(__file__)))'

# scripts that already use relative path (extract_backbone.py)
SKIP = {"extract_backbone.py"}

for fname in os.listdir(CLOUD):
    if not fname.endswith(".py") or fname in SKIP:
        continue
    fpath = os.path.join(CLOUD, fname)
    with open(fpath, encoding="utf-8") as f:
        content = f.read()

    original = content
    # Replace hardcoded ROOT assignment
    content = re.sub(
        r'ROOT\s*=\s*r?"[^"]*idea-x7167-0-1-x7684-paper"?',
        f'ROOT = {NEW_ROOT}',
        content
    )
    # Replace deepfashion_inshop.parquet with deepfashion_subset.parquet
    content = content.replace("deepfashion_inshop.parquet", "deepfashion_subset.parquet")

    if content != original:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"fixed {fname}")
    else:
        print(f"no change {fname}")

print("done")
