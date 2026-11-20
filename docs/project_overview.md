# Project overview

## Aim
Build and evaluate a compact Qwen3-architecture language model for a focused coding-instruction task using CodeDevX/Vibe-Coding-Instruct.

## Key technical distinction
The model is initialised from a new Qwen3Config with random weights. The Qwen/Qwen3-8B tokenizer is used, but pretrained Qwen3-8B weights are not loaded.

## Pipeline
Dataset loading → quality/diversity audit → scenario-grouped split → instruction/response formatting → tokenisation with response-only labels → model training → evaluation → Hugging Face export → optional GGUF/Ollama deployment.

## Scope
This is an experimental student SLM project. The audited data has very low template diversity, so broad coding ability is not established.
