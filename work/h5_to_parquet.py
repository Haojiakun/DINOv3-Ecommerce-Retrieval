"""Convert Fashion-Gen .h5 to .parquet (image bytes + metadata).

Usage:
    work\venv\Scripts\python.exe work\h5_to_parquet.py
"""
import io
import os

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

H5_PATH = r"D:\fashion\fashiongen_256_256_train.h5"
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "work", "fashiongen_train.parquet")
MAX_IMAGES = 10000  # start small; set to None for full 260k


def to_str(b):
    if isinstance(b, bytes):
        return b.decode("utf-8", errors="ignore").strip()
    return str(b).strip()


def main():
    f = h5py.File(H5_PATH, "r")
    n = f["input_image"].shape[0]
    if MAX_IMAGES:
        n = min(n, MAX_IMAGES)
    print(f"h5 total: {f['input_image'].shape[0]}, converting: {n}")

    imgs = f["input_image"]
    pids = f["input_productID"]
    cats = f["input_category"]
    subcats = f["input_subcategory"]
    genders = f["input_gender"]
    brands = f["input_brand"]
    names = f["input_name"]
    descs = f["input_description"]
    poses = f["input_pose"]

    image_bytes = []
    item_id = []
    category = []
    subcategory = []
    gender = []
    brand = []
    name = []
    description = []
    pose = []

    for i in range(n):
        # image -> PNG bytes
        arr = imgs[i]  # (256, 256, 3) uint8
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes.append(buf.getvalue())

        item_id.append(to_str(pids[i]))
        category.append(to_str(cats[i]))
        subcategory.append(to_str(subcats[i]))
        gender.append(to_str(genders[i]))
        brand.append(to_str(brands[i]))
        name.append(to_str(names[i]))
        description.append(to_str(descs[i]))
        pose.append(to_str(poses[i]))

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}")

    table = pa.table({
        "image": pa.array(image_bytes),
        "item_id": pa.array(item_id),
        "category": pa.array(category),
        "subcategory": pa.array(subcategory),
        "gender": pa.array(gender),
        "brand": pa.array(brand),
        "name": pa.array(name),
        "description": pa.array(description),
        "pose": pa.array(pose),
    })
    pq.write_table(table, OUT_PATH)
    print(f"saved {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024 / 1024:.1f} MB)")
    print(f"  rows: {table.num_rows}, columns: {table.column_names}")


if __name__ == "__main__":
    main()
