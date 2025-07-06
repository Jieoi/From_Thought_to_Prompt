"""
consolidate_original_prompts_for_enhancement.py

Purpose:
This script reads 200 `.txt` files containing the original prompts used in your evaluation.
It extracts and compiles them into a single CSV file sorted by numeric ID, to be used as the reference for prompt enhancement and image generation.
"""

import os
import pandas as pd

# === STEP 1: Configuration ===
# Define input/output paths and ensure output folder exists
original_folder = "ORIGINAL"  # Folder containing .txt files named 1.txt to 200.txt
output_csv = "model_outputs/ORIGINAL_prompts.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# === STEP 2: Extract and sort prompts from .txt files ===
# Read all .txt files, extract text, assign numeric ID, sort, and save to CSV
data = []
txt_files = sorted([f for f in os.listdir(original_folder) if f.endswith(".txt")])

for fname in txt_files:
    file_id = os.path.splitext(fname)[0]
    txt_path = os.path.join(original_folder, fname)

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        data.append({"id": file_id, "prompt_original": prompt})
    except Exception as e:
        print(f"[{file_id}] Error reading file: {e}")

df = pd.DataFrame(data)
df["id"] = df["id"].astype(str).str.extract(r"(\d+)").astype(int)
df = df.sort_values(by="id").reset_index(drop=True)

df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} sorted prompts to: {output_csv}")
