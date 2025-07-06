# From Thought to Prompt: Enhancing Text-to-Image Intent Alignment for Faithful T2I Generation

## Overview

This repository presents code, data and documentation for a study on automatic enhancement of user-written prompts for text-to-image (T2I) generation models (e.g., Stable Diffusion). We frame prompt enhancement as a controlled semantic rewriting task that enriches lexical detail while preserving the original intent. To evaluate performance, we introduce the **Enhanced Semantic Expression Score (ESES)**—a composite metric that captures both semantic faithfulness and lexical richness.

## Features

- **Prompt Enhancement Models**  
  - T5-base (~220 M parameters)  
  - BART-base (~406 M parameters)  
  - Qwen-1.5B (instruction-tuned, ~1.5 B parameters)  
  - DeepSeek-Coder-1.3B (code-oriented, ~1.3 B parameters)

- **Evaluation Framework**  
  - **ESES**: Composite text-to-text metric balancing faithfulness and richness  
  - **LLM-Based Ranking**: Proxy “manual” rankings via GPT-4o-mini  
  - **Image-to-Image Metrics**: CLIP cosine similarity, LPIPS  
  - **Text-to-Image Metrics**: CLIP interpolation score, Davidsonian Scene Graph (DSG) ratio

## Webdemo

A static webpage capable of enhancing the prompts and generate new image is avaliable at: https://drive.google.com/drive/folders/1OU19AtYCvr_IaP539iadXs1Vs9Zr4IHq

Note that NSFW feature has been turned OFF. 

## Authors
- Bryan Chang Tze Kin: tk.chang.2024@mitb.smu.edu.sg
- Dau Vu Dang Khoi: vdk.dau.2024@mitb.smu.edu.sg
- Li Xinjie ✉ : xinjie.li.2024@mitb.smu.edu.sg
- Nguyen Nguyen Ha: ha.nguyen.2024@mitb.smu.edu.sg
- Samuel Lee Wei Sheng: ws.lee.2024@mitb.smu.edu.sg


## Acknowledgements

We would like to express our sincere gratitude to **Prof. Deng Yang** (ydeng@smu.edu.sg) for his invaluable guidance and support throughout this project. We are honored to be the first-ever cohort he has taught.
