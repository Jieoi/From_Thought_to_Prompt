From Thought to Prompt: Enhancing Text-to-Image Intent Alignment for Faithful T2I Generation
Overview
This repository contains the code, data, and documentation for our project that investigates automatic enhancement of user-written prompts for text-to-image (T2I) generation models such as Stable Diffusion. We frame prompt enhancement as a controlled semantic rewriting task that enriches lexical detail while preserving original intent. The project also introduces the Enhanced Semantic Expression Score (ESES), a novel composite metric capturing both semantic faithfulness and lexical richness 
.

Features
Prompt Enhancement Models: Fine-tuned variants of

T5-base (~220M parameters)

BART-base (~406M parameters)

Qwen-1.5B (instruction-tuned, ~1.5B parameters)

DeepSeek-Coder-1.3B (code-oriented, ~1.3B parameters)

Evaluation Framework:

ESES: Composite text-to-text metric balancing faithfulness and richness

LLM-Based Ranking: Proxy “manual” rankings via GPT-4o-mini

Image-to-Image Metrics: CLIP cosine similarity, LPIPS

Text-to-Image Metrics: CLIP interpolation score, Davidsonian Scene Graph (DSG) ratio