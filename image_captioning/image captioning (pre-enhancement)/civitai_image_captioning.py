# === CIVITAI IMAGE CAPTIONING PIPELINE ===
"""
This script processes a folder of images and their accompanying metadata (JSON files) scraped from Civitai.
It performs the following:
1. Extracts text prompts from metadata.
2. Sends images to OpenAI's GPT-4o-mini for captioning.
3. Skips images that are corrupt or unreadable.
4. Produces a final CSV that includes: image ID, prompt, image filename, and generated caption.

Final output: civitai_image_prompt_captioned_cleaned.csv
"""

import os
import json
import base64
import time
import pandas as pd
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI, RateLimitError

# === STEP 0: CONFIGURATION ===
# Load API key from .env file and set up paths
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

TARGET_DIR = ""
FINAL_OUTPUT_CSV = os.path.join(TARGET_DIR, "civitai_image_prompt_captioned_cleaned.csv")

# === STEP 1: EXTRACT PROMPTS FROM JSON ===
# This step reads all .json files, checks for corresponding .jpg files,
# extracts non-empty and unique prompts, and returns a DataFrame.
def extract_prompts():
    seen_prompts = set()
    rows = []

    for filename in os.listdir(TARGET_DIR):
        if not filename.endswith(".json"):
            continue

        base = filename[:-5]
        json_path = os.path.join(TARGET_DIR, filename)
        jpg_path = os.path.join(TARGET_DIR, base + ".jpg")

        if not os.path.exists(jpg_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            prompt = data.get("meta", {}).get("prompt", "").replace('"', '""').strip()

        if not prompt or prompt in seen_prompts:
            continue

        seen_prompts.add(prompt)
        rows.append([base, prompt, base + ".jpg"])

    df = pd.DataFrame(rows, columns=["id", "prompt", "image_filename"])
    return df

# === STEP 2: ENCODE IMAGE TO BASE64 ===
# Converts image to base64 PNG format for API submission.
# Returns None if image is unreadable or corrupt.
def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return None

# === STEP 3: CAPTION IMAGES AND FILTER BAD ONES ===
# Iterates through the DataFrame, encodes each image, sends it to OpenAI,
# retrieves a caption, and filters out any corrupted images.
# Final CSV is written after excluding all failed or unreadable images.
def generate_captions(df):
    records = []
    fail_count = 0
    max_failures = 10
    bad_files = set()

    print(f"Starting captioning for {len(df)} images...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_file = row["image_filename"]
        image_path = os.path.join(TARGET_DIR, image_file)

        b64_image = encode_image(image_path)
        if not b64_image:
            records.append([row["id"], row["prompt"], image_file, None])
            bad_files.add(image_file)
            fail_count += 1
            if fail_count >= max_failures:
                print("Too many failures. Exiting early.")
                break
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
            time.sleep(16)  # Rate limit buffer
        except RateLimitError:
            time.sleep(5)
            continue
        except Exception:
            records.append([row["id"], row["prompt"], image_file, None])
            bad_files.add(image_file)
            fail_count += 1
            if fail_count >= max_failures:
                break

    df_out = pd.DataFrame(records, columns=["id", "prompt", "image_filename", "caption"])
    df_out = df_out[~df_out["image_filename"].isin(bad_files)]
    df_out.to_csv(FINAL_OUTPUT_CSV, index=False)
    print(f"Saved cleaned captioned CSV: {FINAL_OUTPUT_CSV}")

# === MAIN EXECUTION ===
# Extract prompts from metadata and generate captions.
if __name__ == "__main__":
    df_prompts = extract_prompts()
    generate_captions(df_prompts)