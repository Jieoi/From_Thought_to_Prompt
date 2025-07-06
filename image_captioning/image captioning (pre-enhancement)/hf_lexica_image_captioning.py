# === LEXICA DATASET: IMAGE-PROMPT-CAPTION PIPELINE ===
"""
This script builds a clean, captioned dataset using images and prompts from the Lexica Stable Diffusion v1.5 dataset.
https://huggingface.co/datasets/yuwan0/lexica-stable-diffusion-v1-5

It performs the following steps:

1. Extracts prompts and image references from `.parquet` metadata files.
2. Downloads images as `.jpg` or `.png` files and saves them locally.
3. Filters out corrupted or unreadable images.
4. Sends valid images to OpenAI's GPT-4o-mini to generate one-line captions.
5. Supports automatic resume by checking existing output and skipping completed entries.
6. Exports the final dataset to a CSV file with the following columns:
   id, prompt, image_filename, caption
"""

import os
import base64
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# === STEP 0: CONFIGURATION ===
# Load API key and define folders and file paths
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

data_folder = ""
image_folder = ""
output_csv = ""
max_consecutive_fails = 20
max_api_failures = 10

os.makedirs(image_folder, exist_ok=True)

# === STEP 1: EXTRACT PROMPTS & DOWNLOAD IMAGES FROM PARQUET ===
# Parses each .parquet file to collect prompt + image reference, then downloads valid images
# Skips corrupted or unsupported formats, and validates that images are readable

def extract_and_download():
    records = []
    id_counter = 1
    consecutive_fails = 0
    parquet_files = [f for f in os.listdir(data_folder) if f.endswith(".parquet")]

    for file in parquet_files:
        full_path = os.path.join(data_folder, file)
        print(f"\nProcessing {file}...")

        try:
            df = pd.read_parquet(full_path)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        if df.empty:
            print(f"Skipping {file} (empty DataFrame)")
            continue

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Rows in {file}"):
            try:
                prompt = row.get("text") or row.get("prompt")
                img_data = row.get("image") or row.get("url")

                if not prompt or not img_data:
                    continue

                img_id = f"img_{id_counter:07d}"
                img_ext = "jpg"
                img_bytes = None

                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_bytes = img_data["bytes"]
                    img_ext = "png"

                elif isinstance(img_data, dict) and "url" in img_data:
                    img_url = img_data["url"]
                    if isinstance(img_url, str):
                        img_ext = img_url.split('.')[-1].split('?')[0][:4].lower()
                        if img_ext not in ["jpg", "jpeg", "png", "webp"]:
                            continue
                        response = requests.get(img_url, timeout=10)
                        response.raise_for_status()
                        img_bytes = response.content

                elif isinstance(img_data, str):
                    img_url = img_data
                    img_ext = img_url.split('.')[-1].split('?')[0][:4].lower()
                    if img_ext not in ["jpg", "jpeg", "png", "webp"]:
                        continue
                    response = requests.get(img_url, timeout=10)
                    response.raise_for_status()
                    img_bytes = response.content

                else:
                    continue

                if img_bytes:
                    img_filename = f"{img_id}.{img_ext}"
                    img_path = os.path.join(image_folder, img_filename)

                    if os.path.exists(img_path):
                        id_counter += 1
                        continue

                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)

                    try:
                        with Image.open(img_path) as test_img:
                            test_img.verify()
                    except Exception:
                        os.remove(img_path)
                        continue

                    records.append({
                        "id": img_id,
                        "prompt": prompt,
                        "image_filename": img_filename
                    })

                    id_counter += 1
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1

            except Exception:
                consecutive_fails += 1
                if consecutive_fails >= max_consecutive_fails:
                    print(f"Aborting: {max_consecutive_fails} consecutive failures.")
                    break

        if consecutive_fails >= max_consecutive_fails:
            break

    return pd.DataFrame(records)

# === STEP 2: ENCODE IMAGE TO BASE64 ===
# Converts a valid image into a base64-encoded PNG string

def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return None

# === STEP 3: GENERATE IMAGE CAPTIONS USING GPT-4o ===
# Checks for existing captioned rows to resume automatically.
# Sends base64-encoded image to GPT-4o-mini and stores the caption.

def caption_images(df):
    existing_captions = set()
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        existing_captions = set(df_existing["image_filename"].dropna())
        df = df[~df["image_filename"].isin(existing_captions)]
        df_existing = df_existing.dropna(subset=["caption"])
    else:
        df_existing = pd.DataFrame(columns=["id", "prompt", "image_filename", "caption"])

    if df.empty:
        print("All images already captioned.")
        return

    records = []
    fail_count = 0

    print(f"Starting captioning for {len(df)} images...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_file = row["image_filename"]
        image_path = os.path.join(image_folder, image_file)

        if not os.path.exists(image_path):
            continue

        b64_image = encode_image(image_path)
        if not b64_image:
            fail_count += 1
            continue

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this image briefly, including the subject(s) and visual details. Use one clear sentence."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                    ]}
                ],
                max_tokens=100
            )
            caption = response.choices[0].message.content.strip()
            records.append([row["id"], row["prompt"], image_file, caption])
            fail_count = 0

        except Exception:
            fail_count += 1
            if fail_count >= max_api_failures:
                print("Too many consecutive API failures. Stopping early.")
                break

    df_new = pd.DataFrame(records, columns=["id", "prompt", "image_filename", "caption"])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(output_csv, index=False)
    print(f"\nSaved final captioned dataset to: {output_csv}")

# === STEP 4: RUN FULL PIPELINE ===
# Orchestrates the process from extraction to captioning to export
if __name__ == "__main__":
    df = extract_and_download()
    if df.empty:
        print("No valid images extracted.")
    else:
        caption_images(df)