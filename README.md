# Vibe Coding Assistant — Strict, Dataset-Only Edition (Qwen3-1.7B)

A local terminal chatbot for **vibe coding in general** (coding, debugging, building apps,
websites and AI models, deployment) that also uses your 14-row dataset as reference.
Anything outside vibe coding ("teach me French", "make coffee", jokes, math...) is politely refused.

## How it works (every message is checked on its own)

1. **Greetings** ("hi", "thanks", "who are you") get a friendly canned reply.
2. **Scope gate in code** — the prompt must contain a coding / software / AI-building term
   (python, React, API, bug, app, deploy, AI model, LLM, Ollama...), or strongly match a dataset
   row, or be a short follow-up ("explain that more") in an on-topic chat. Otherwise the app prints
   a fixed refusal. **The model is not called for refused prompts.**
3. **On-topic** prompts go to Qwen3-1.7B with the best matching dataset rows as context.
4. After a refusal the app is immediately ready: the next message is checked from scratch,
   and refused messages are not stored in the chat history.
5. A strict system prompt (app + `Modelfile`) backs this up for borderline prompts.

Why not fine-tune? 14 unique rows are far too few to teach reliable refusals; a code-level gate is
deterministic.

## Setup (Windows, VS Code terminal)

```powershell
ollama pull qwen3:1.7b   # skip if already downloaded
ollama create vibe-coding-assistant -f .\Modelfile   # re-run: rebuilds the assistant with the strict rules
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
python test_retrieval.py     # no Ollama needed; checks retrieval + gate
python app.py
```

Keep `vibe_dataset_examples.json` next to `app.py`.

After replacing the files, **re-run `ollama create ...`** so the new Modelfile rules take effect.

## Controls
Enter = send · Ctrl+J = new line · Ctrl+N = new chat · Ctrl+L = clear · Ctrl+C = quit

## Customizing the scope
In `app.py`:
- `SCOPE_WORDS` / `SCOPE_PHRASES` — add keywords (e.g. "unity", "flutter") to allow more topics.
  Remove a word to be stricter.
- `REFUSAL_MESSAGE`, `GREETING_MESSAGE` — the texts shown.
- `STRONG_RETRIEVAL_SCORE` — lower = more permissive.

## Known limits
- The gate is keyword-based: a valid coding question that uses none of the known words may be refused
  (add the word to `SCOPE_WORDS`), and an off-topic question that happens to contain one
  (e.g. "app") may pass; the system prompt is the backup then.
- Mixed prompts ("explain React and teach me French") pass the gate; the system prompt
  handles those but a 1.7B model is not perfect.
- Conversation history is in memory only. The app does not run code or edit files.
