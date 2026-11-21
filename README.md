# Vibe Coding SLM & Local Assistant

A local **Vibe Coding AI assistant** built around Qwen3 and the professor-assigned `CodeDevX/Vibe-Coding-Instruct` dataset. The repository documents the complete workflow: dataset collection and audit, preprocessing, compact-model training, evaluation, Hugging Face publishing, GGUF export, Ollama deployment, and the local chatbot application.

> **Important model distinction:** This project contains two related but different model components. The original academic experiment is a **173.4M-parameter Qwen3 model initialized from random weights and trained for 1,200 optimizer updates**. The conversational assistant uses the **pretrained `qwen3:1.7b` model through Ollama**, with the same assigned dataset connected as a small retrieval knowledge base. Do not describe the pretrained 1.7B assistant as the scratch-trained 173.4M model.

---

## Table of contents

1. [Project overview](#project-overview)
2. [System architecture](#system-architecture)
3. [Repository structure](#repository-structure)
4. [Technology stack](#technology-stack)
5. [Requirements](#requirements)
6. [Dataset collection](#1-dataset-collection)
7. [Dataset audit and preprocessing](#2-dataset-audit-and-preprocessing)
8. [Compact Qwen3 model training](#3-compact-qwen3-model-training)
9. [Evaluation and known results](#4-evaluation-and-known-results)
10. [Hugging Face publishing](#5-hugging-face-publishing)
11. [GGUF conversion and quantization](#6-gguf-conversion-and-quantization)
12. [Ollama deployment: scratch-trained model](#7-ollama-deployment-scratch-trained-model)
13. [Local conversational assistant: Qwen3 1.7B](#8-local-conversational-assistant-qwen3-17b)
14. [Strict terminal edition](#9-strict-terminal-edition)
15. [Running the assistant locally](#10-running-the-assistant-locally)
16. [Testing checklist](#11-testing-checklist)
17. [GitHub publishing](#12-publishing-the-project-to-github)
18. [Troubleshooting](#troubleshooting)
19. [Limitations and responsible reporting](#limitations-and-responsible-reporting)
20. [References](#references)

---

## Project overview

The project explores how a compact Qwen3 architecture can be trained using a Vibe Coding instruction dataset and how that knowledge can be made useful in a local assistant.

The work has two parts:

### A. Original scratch-trained SLM experiment

A compact Qwen3 causal language model was created using `Qwen3Config` and `Qwen3ForCausalLM`. Its parameters were initialized randomly; pretrained Qwen3 language-model weights were **not** loaded. The Qwen3-8B tokenizer was used to tokenize the professor-assigned dataset.

This experiment demonstrates the model-construction, data-preparation, training, evaluation, export and quantization pipeline. It is retained as the project's original academic baseline.

### B. Local Vibe Coding assistant

A separate local assistant uses the pretrained `qwen3:1.7b` model as its language foundation. The professor-assigned dataset is loaded locally and relevant examples are retrieved at request time. The assistant also uses a small recent-message memory and structured task-planning instructions.

This is the component intended to provide a more capable, conversational demonstration. Its general language ability comes from the pretrained Qwen3 1.7B model; the dataset retrieval component supplies additional project-specific context.

### Main capabilities of the local assistant

- Normal conversation and follow-up questions.
- Coding and software-development explanations.
- Structured task and implementation planning.
- Retrieval of relevant examples from the assigned Vibe Coding dataset.
- Short conversation memory for recent exchanges.
- Thinking on complex requests, while showing the user only the final answer.
- Local operation through Ollama after the one-time model and dependency setup.

The assistant is a planning and conversational prototype. It does not automatically modify project files, execute terminal commands, or run generated code.

---

## System architecture

### Hybrid conversational assistant

```text
                   User
                    |
                    v
          Local chatbot interface
          (Gradio / localhost)
                    |
                    v
             Request handling
                    |
          +---------+----------+
          |                    |
          v                    v
   Recent conversation    Dataset retrieval
       memory              (TF-IDF-style)
   last 3 exchanges       top 2 examples
          |                    |
          +---------+----------+
                    |
                    v
          Structured task-planning
               instructions
                    |
                    v
          Pretrained Qwen3 1.7B
              through Ollama
                    |
            +-------+-------+
            |               |
            v               v
        Thinking         Final answer
        kept hidden      shown to user
```

The retrieval and memory components are intentionally lightweight. The prepared dataset has only a small number of unique mappings, so this design does not require a separate embedding model, vector database, LangChain, FAISS or ChromaDB.

### Original training and deployment pipeline

```text
Hugging Face dataset
        |
        v
Dataset inspection and audit
        |
        v
Normalization and deduplication
        |
        v
Train / validation / test preparation
        |
        v
Qwen3 tokenizer and response-only labels
        |
        v
Randomly initialized compact Qwen3 model
        |
        v
1,200 optimizer updates
        |
        v
Evaluation and best-checkpoint selection
        |
        v
Hugging Face model export
        |
        v
GGUF conversion and quantization
        |
        v
Ollama import and local inference
```

---

## Repository structure

The following is the recommended layout for keeping the training experiment, assistant application and deployment files organized. Keep the two model tracks in separate folders so their checkpoints and Modelfiles cannot be mixed up.

```text
Vibe-Coding-SLM/
|
|-- README.md
|-- .gitignore
|
|-- notebooks/
|   |-- Vibe_Coding_Qwen3_SLM_End_to_End.ipynb
|   `-- 5_slm_from_scratch_qwen.ipynb       # professor's reference notebook
|
|-- training/
|   `-- export_dataset_examples_colab.py
|
|-- reports/
|   |-- dataset_audit.json
|   |-- training_run.json
|   |-- training_history.json
|   |-- evaluation_results.json
|   `-- model_metadata.json
|
|-- assistant/
|   |-- app.py                              # full local assistant interface
|   |-- Modelfile                           # based on qwen3:1.7b
|   |-- requirements.txt
|   `-- vibe_dataset_examples.json
|
|-- strict_terminal/
|   |-- app.py                              # optional scope-gated terminal app
|   |-- Modelfile
|   |-- requirements.txt
|   |-- test_retrieval.py
|   `-- vibe_dataset_examples.json
|
|-- ollama_scratch_model/
|   |-- Modelfile.Q8_0
|   |-- install_model.ps1
|   |-- run_demo.bat
|   |-- README_WINDOWS_LOCAL.md
|   `-- vibe-coding-qwen3-Q8_0.gguf         # keep large model files outside Git if needed
|
`-- data/
    `-- README.md                           # dataset source and preparation notes
```

File names can be adjusted to match the actual files in the release. The key rule is to keep the **scratch-trained model deployment** and **pretrained Qwen3 1.7B assistant** clearly separated.

---

## Technology stack

| Area | Technology |
|---|---|
| Dataset hosting and collection | Hugging Face Datasets |
| Training framework | PyTorch and Hugging Face Transformers |
| Model architecture | Compact Qwen3 causal language model |
| Tokenizer | Qwen3-8B tokenizer |
| Dataset retrieval | Lightweight TF-IDF-style text matching |
| Local model runtime | Ollama |
| Conversational assistant model | Pretrained Qwen3 1.7B |
| Local interface | Gradio for the browser-based assistant; terminal interface for the strict edition |
| Development environment | Python, VS Code and Windows 11 |
| Model export | Safetensors / Hugging Face format and GGUF |

---

## Requirements

### Training environment

The original training workflow was designed to run in Google Colab using a GPU runtime. The recorded compact-model experiment used 1,200 optimizer updates. The professor's original notebook is retained as an architecture and workflow reference; its TinyStories dataset and larger training configuration are not substituted for the assigned Vibe Coding dataset.

### Local inference environment

The local development and demo machine recorded for this project is:

- Windows 11
- NVIDIA GTX 1650 Laptop GPU with 4 GB VRAM
- 8 GB system RAM
- Python installed and available through the `py` launcher
- VS Code
- Ollama for Windows

The Q8_0 export of the scratch-trained model is approximately 181.4 MB. The separate pretrained Qwen3 1.7B model is substantially larger and is downloaded once through Ollama. Actual GPU/CPU placement and generation speed depend on the installed driver, available memory and other applications running on the laptop.

---

## 1. Dataset collection

The assigned source dataset is:

**[CodeDevX/Vibe-Coding-Instruct](https://huggingface.co/datasets/CodeDevX/Vibe-Coding-Instruct)**

The dataset is collected from the Hugging Face Hub using the `datasets` library. The source dataset is retained as assigned by the professor. No replacement dataset or additional training dataset is introduced in this workflow.

### Install the dataset library

```bash
pip install datasets
```

### Load the original dataset in Colab or Python

```python
from datasets import load_dataset

DATASET_ID = "CodeDevX/Vibe-Coding-Instruct"
raw_dataset = load_dataset(DATASET_ID)

print(raw_dataset)

for split_name, split_data in raw_dataset.items():
    print(f"\nSplit: {split_name}")
    print("Rows:", len(split_data))
    print("Columns:", split_data.column_names)
    print("Features:", split_data.features)
    print("First example:", split_data[0])
```

The inspected source contained one `train` split with approximately 1.1 million records and the columns:

- `instruction`
- `input`
- `output`
- `prompt`

The raw dataset is much larger than the number of distinct instruction/response patterns it represents. This distinction is important during preprocessing and evaluation.

---

## 2. Dataset audit and preprocessing

### Why an audit is necessary

The dataset audit recorded approximately **1,100,000 source records**, but only **14 distinct instruction/response mappings** after the scenario-number variations were normalized. The `input` field was empty in the inspected examples. Many raw records therefore repeat an underlying mapping while changing only a scenario identifier.

The assigned source dataset is not replaced. Normalization is used only to identify equivalent records, remove duplicate training instances from the compact prepared set, and prevent near-identical examples from leaking across the data splits.

### Consistent scenario normalization

An earlier preprocessing/audit cell used an incorrect double-escaped regular expression. Use one normalization function consistently throughout audit, preparation and evaluation:

```python
import re

def clean_instruction(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(
        r"\bScenario\s+\d+\s*:?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split()).strip()
```

The expression above removes `Scenario 0`, `Scenario 12:` and similar identifiers. An equivalent, simpler expression can also be used:

```python
text = re.sub(r"\bScenario\s+\d+\s*:?", "", text, flags=re.IGNORECASE)
```

When editing this expression, test it against examples both with and without a colon. Do not use a double-backslash pattern such as `r"Scenario\\s+\\d+"` for this purpose.

### Deduplication and split strategy

1. Preserve the downloaded source dataset unchanged.
2. Normalize the scenario number in each instruction to create a stable comparison key.
3. Keep the `input` and `output` values in the key so that different input/output mappings are not merged accidentally.
4. Remove exact duplicate normalized mappings from the compact prepared dataset.
5. Split the unique mappings into training, validation and test groups **after deduplication**.
6. Verify that the normalized instruction/response keys do not overlap between the three splits.
7. Save the prepared splits and the dataset audit report.

The latest prepared split recorded in the project contains:

| Split | Unique examples |
|---|---:|
| Training | 12 |
| Validation | 1 |
| Test | 1 |
| **Total** | **14** |

This is a very small dataset for language-model training. It is retained because it is the assigned project dataset, but the resulting model must not be presented as having learned broad coding knowledge from 1.1 million independent examples.

### Dataset export for the assistant

The local assistant uses a compact JSON file named `vibe_dataset_examples.json`. The export step reads the prepared data from the existing Colab/Drive project, combines the prepared splits, removes duplicate mappings, and writes the examples to JSON. The resulting JSON is placed next to the assistant's `app.py` file.

This enables local retrieval without downloading the full 459 MB source dataset every time the chatbot starts.

---

## 3. Compact Qwen3 model training

The original SLM experiment follows the Qwen3 architecture using the Hugging Face implementation.

### Model configuration

| Parameter | Recorded value |
|---|---:|
| Architecture | Qwen3 causal language model |
| Model class | `Qwen3ForCausalLM` |
| Configuration class | `Qwen3Config` |
| Initialization | Random weights |
| Parameters | 173,404,416 (approximately 173.4M) |
| Transformer layers | 6 |
| Hidden size | 512 |
| Attention heads | 8 |
| Key/value heads | 2 |
| Feed-forward size | 1,536 |
| Context length | 128 tokens |
| Tokenizer | Qwen3-8B tokenizer |
| Training updates | 1,200 optimizer updates |
| Training objective | Response-only loss |

The compact model retains key Qwen3-style architectural components, including RMSNorm, Grouped Query Attention, QK normalization, SwiGLU and RoPE. The implementation uses the Hugging Face Qwen3 model classes rather than manually reimplementing every transformer layer.

### Important initialization detail

The training notebook creates the model from a configuration, conceptually:

```python
config = Qwen3Config(**MODEL_CONFIG)
model = Qwen3ForCausalLM(config)
```

This initializes new model weights. It does **not** load the pretrained weights of `Qwen/Qwen3-8B`. The pretrained Qwen3-8B tokenizer is used, but a tokenizer is not the same thing as pretrained model weights.

### Training strategy

- The training prompt is formatted consistently with the notebook's instruction/input/response structure.
- Loss is applied to the response tokens, while instruction and input tokens are masked from the loss.
- The training configuration is capped at 1,200 optimizer updates.
- Training, validation and checkpoint information are saved as reports.
- The best validation checkpoint is retained separately from the latest checkpoint.

**Checkpoint note:** The latest inspected notebook reports the best checkpoint at update 100, with a validation loss of 4.8824. The final training loop completed 1,200 updates, but the best checkpoint is not necessarily the checkpoint from the final update. Use the reported best checkpoint for evaluation/export unless a later verified run produces different results.

If training must genuinely resume from a checkpoint, the checkpoint must be loaded **before** entering the training loop. Restore the model, optimizer, scheduler, mixed-precision scaler, current update number and relevant training history before continuing. A cell that reloads a checkpoint only after the training loop has ended restores a model for inspection; it does not resume training.

### Mixed precision

For a Tesla T4 Colab runtime, use FP16 rather than assuming BF16 acceleration is available. A suitable future-run configuration is:

```python
USE_BF16 = False
AMP_DTYPE = torch.float16
USE_SCALER = True
```

This is a training-environment setting. It does not change the model's architecture or the separate Windows inference workflow.

---

## 4. Evaluation and known results

The latest inspected training diagnostics reported:

| Metric | Recorded result |
|---|---:|
| Raw source rows | 1,100,000 |
| Unique instruction/response mappings | 14 |
| Training examples after deduplication | 12 |
| Validation examples | 1 |
| Test examples | 1 |
| Model parameters | 173,404,416 |
| Optimizer updates completed | 1,200 |
| Best validation checkpoint | Update 100 |
| Best validation loss | 4.8824 |
| Held-out exact-match accuracy | 0% (one test example) |

The training loss approached zero while validation loss worsened. This indicates overfitting to the tiny training set. The held-out test set contains only one unique example, so 0% exact match is a warning about this run, not a statistically reliable estimate of performance on general coding tasks.

An earlier experiment reported different evaluation figures. Do not combine metrics from different notebook runs. For any final presentation or research paper, use the metrics from the exact checkpoint being demonstrated, and identify the run using its saved `training_run.json`, `training_history.json` and `evaluation_results.json` files.

The scratch-trained 173.4M model was successfully imported into Ollama using its Q8_0 GGUF export, but the inspected local outputs were incoherent. Successful GGUF conversion or Ollama import confirms that the artifact loads; it does not by itself demonstrate useful language-generation quality.

---

## 5. Hugging Face publishing

The project uses Hugging Face to host the trained model and the prepared version of the assigned dataset.

Previously configured repository IDs in the project were:

- Model: [`Aslam7/Vibe-Coding-Qwen3-SLM`](https://huggingface.co/Aslam7/Vibe-Coding-Qwen3-SLM)
- Prepared dataset: [`Aslam7/Vibe-Coding-Instruct-Prepared`](https://huggingface.co/datasets/Aslam7/Vibe-Coding-Instruct-Prepared)

Use the actual repository IDs configured in the latest notebook if they have since changed.

### Authenticate securely

Install the Hub library and authenticate in the environment where the upload is performed:

```bash
pip install -U huggingface_hub
hf auth login
```

Use an access token with the required permissions. Never hard-code the token into a notebook, Python file, README, GitHub commit or screenshot.

### Upload the model directory

After exporting the selected model and its required tokenizer/configuration files to `hf_model/`, upload that folder to the model repository:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="hf_model",
    repo_id="Aslam7/Vibe-Coding-Qwen3-SLM",
    repo_type="model",
)
```

### Upload the prepared dataset

After saving the prepared train/validation/test splits and dataset documentation to `hf_dataset/`, upload them to the dataset repository:

```python
api.upload_folder(
    folder_path="hf_dataset",
    repo_id="Aslam7/Vibe-Coding-Instruct-Prepared",
    repo_type="dataset",
)
```

The original source dataset remains attributed to `CodeDevX/Vibe-Coding-Instruct`. The prepared repository should explain the normalization, deduplication and split process rather than implying that it contains 1.1 million distinct examples.

### Recommended model repository contents

- Model configuration (`config.json`).
- Model weights (`model.safetensors`, or the relevant sharded files).
- Required tokenizer files (`tokenizer.json`, `tokenizer_config.json`, vocabulary/merge files where applicable, and special-token files).
- Model card (`README.md`).
- Model metadata and evaluation report.

After repairing tokenizer files for GGUF conversion, update the Hugging Face model repository as well so the uploaded model and GGUF export use consistent tokenizer assets.

---

## 6. GGUF conversion and quantization

GGUF files are generated from the exported Hugging Face model using `llama.cpp` conversion and quantization tools. The tokenizer assets must be complete and compatible before conversion.

A tokenizer-related conversion issue was previously identified. The repair workflow used the original Qwen3 tokenizer assets and validated the tokenizer configuration before converting the trained weights. In this project, the vocabulary length used for validation was reported as 151,669; do not compare it against a smaller base-vocabulary number without accounting for added tokens.

### Quantized files recorded in the project

| Quantization | Approximate size | Use |
|---|---:|---|
| Q8_0 | 181.4 MB | Preferred local demo artifact for the scratch-trained model |
| Q5_K_M | 129.4 MB | Smaller alternative |
| Q4_K_M | 118.2 MB | Smallest recorded alternative |

The recommended scratch-trained model file for the Windows Ollama demo is:

```text
vibe-coding-qwen3-Q8_0.gguf
```

Keep the F16 intermediate and quantized outputs in the project's `gguf/` folder. Rebuild the final release package **after** GGUF conversion and Modelfile creation so that the ZIP actually contains the files it claims to contain.

---

## 7. Ollama deployment: scratch-trained model

This section imports the **173.4M scratch-trained model** into Ollama. It is separate from the Qwen3 1.7B conversational assistant described later.

### Step 1 — Install and start Ollama

Install Ollama for Windows and make sure its local service is running. The local API is normally available at:

```text
http://127.0.0.1:11434
```

### Step 2 — Place the GGUF and Modelfile

Place `vibe-coding-qwen3-Q8_0.gguf` in the path expected by `Modelfile.Q8_0`. A typical deployment folder is:

```text
ollama_scratch_model/
|-- Modelfile.Q8_0
|-- vibe-coding-qwen3-Q8_0.gguf
|-- install_model.ps1
|-- run_demo.bat
`-- app.py
```

The `Modelfile.Q8_0` must reference the actual GGUF filename and preserve the prompt format used by the training notebook. Do not apply an additional chat template on top of the training prompt. The original local demo used Ollama's raw prompt mode for this reason.

### Step 3 — Create the Ollama model

From PowerShell in the deployment folder:

```powershell
ollama create vibe-coding-qwen3 -f .\Modelfile.Q8_0
```

### Step 4 — Confirm and run

```powershell
ollama list
ollama run vibe-coding-qwen3
```

Example prompts that correspond to the assigned dataset include:

```text
Create a coding assistant. Provide a practical implementation plan.
Add AI chat to an app. Provide a practical implementation plan.
Deploy a MERN application. Provide a practical implementation plan.
React app re-renders infinitely. Provide a practical implementation plan.
```

The original training data is single-turn and the model context length is 128 tokens. The Windows demo therefore sends each generation as an independent request; any transcript shown by the basic demo is display history, not persistent model-side conversation memory.

### Step 5 — Run the local demo interface

If the Windows demo files are present, launch the interface with:

```powershell
python app.py
```

Or use the provided `run_demo.bat`. The interface communicates with Ollama through localhost and does not require a hosted inference API.

### Step 6 — Check local execution

While a response is generating, open another PowerShell window:

```powershell
ollama ps
```

For NVIDIA diagnostics, if the driver tools are installed:

```powershell
nvidia-smi
```

These commands help determine whether Ollama is using the GPU, CPU or a combination. Do not assume that all inference is running on the GPU; check the actual runtime report.

---

## 8. Local conversational assistant: Qwen3 1.7B

The full conversational assistant is a separate application. It uses the pretrained Qwen3 1.7B model through Ollama and connects the professor-assigned dataset at runtime. It does **not** fine-tune the 1.7B model on the 14 mappings.

### Why this assistant is separate

The 173.4M training experiment starts from random weights and has only 14 unique instruction/response mappings available after deduplication. That experiment is useful as the academic baseline, but it cannot provide broad pretrained language ability from those examples alone.

The local assistant instead starts with pretrained Qwen3 1.7B for normal language understanding and uses the assigned dataset as additional retrieved context. This keeps the original academic experiment intact while providing a better conversational demonstration.

### Assistant-side components

| Component | Lightweight configuration |
|---|---|
| Language model | Pretrained `qwen3:1.7b` through Ollama |
| Dataset retrieval | TF-IDF-style matching |
| Retrieved context | At most 2 relevant dataset examples |
| Conversation memory | Last 3 exchanges |
| Task planning | Structured system instructions |
| Thinking | Enabled for complex requests only |
| Thinking display | Show a short “Thinking…” status; hide the internal thinking text |
| Context window | 2,048 tokens as the initial target |
| Initial response budget | 256 tokens in the initial configuration; complex requests may use a larger configured budget |
| Model keep-alive | 10 minutes |

The retrieval system is intentionally small. Because the prepared knowledge base contains only 14 unique mappings, a lightweight lexical/TF-IDF-style search is sufficient for this prototype. A separate embedding model and vector database are not required.

### Download the pretrained runtime model

```powershell
ollama pull qwen3:1.7b
```

This is a one-time download. After it is stored locally, Ollama can run the model without downloading it again.

### Build the customised assistant model

From the assistant folder, where the `Modelfile` is located:

```powershell
ollama create vibe-coding-assistant -f .\Modelfile
```

Check that it is registered:

```powershell
ollama list
```

A direct model test can be run with:

```powershell
ollama run vibe-coding-assistant
```

### Export the prepared dataset to JSON

The assistant expects:

```text
vibe_dataset_examples.json
```

The export script `training/export_dataset_examples_colab.py` (or the corresponding export cell in the training notebook) should:

1. Read the saved prepared dataset from the existing Drive project.
2. Combine the training, validation and test examples for retrieval use.
3. Remove duplicate normalized mappings.
4. Write the compact `vibe_dataset_examples.json` file.
5. Download the JSON file from Colab.

Place `vibe_dataset_examples.json` in the same folder as the assistant's `app.py` unless the application configuration specifies another path.

### Install and launch the browser-based interface

Open PowerShell in the assistant folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
python app.py
```

The Gradio interface is served locally. The usual local address for the project is:

```text
http://127.0.0.1:7860
```

The application should use Ollama's local chat API, retain only the most recent few exchanges, and retrieve no more than two relevant dataset examples for each request. Streaming can be used to make the interface feel responsive.

### Private thinking, visible final answer

For complex planning requests, the application can ask Qwen3 to think before answering. Ollama separates the model's thinking field from the final content field. The interface should display a small status such as `Thinking...` while generation is running, then display only the final answer. It should not print the internal thinking field into the visible chat transcript.

To keep responses efficient:

- Use thinking only for complex requests.
- Keep the recent conversation history short.
- Retrieve no more than two dataset examples.
- Use a smaller generation budget for greetings and simple questions.
- Allow a larger response budget for multi-stage plans.
- Keep the model loaded for a limited period using the configured keep-alive value.

The interface can provide code blocks and structured plans, but the assistant does not automatically create files or execute generated code.

---

## 9. Strict terminal edition

The repository may also include a **strict, dataset-aware terminal edition** with a deterministic scope gate. This is an optional restricted mode and is not the same user experience as the general conversational Gradio assistant.

### Behaviour

1. **Greetings:** short messages such as `hi`, `thanks` and `who are you` receive friendly predefined replies.
2. **Scope gate in Python:** a request must contain a known coding/software/AI-building term, strongly match a dataset example, or qualify as a short follow-up in an already on-topic conversation.
3. **Out-of-scope requests:** the application returns a fixed refusal. The Ollama model is not called for a rejected request.
4. **On-topic requests:** the request is sent to Qwen3 1.7B with the best matching dataset examples as context.
5. **History handling:** a refused request is not added to the conversation history. The next request is checked independently.
6. **Prompt-level guardrails:** the application instructions and the Ollama `Modelfile` both reinforce the intended scope.

This edition is intended for vibe-coding topics such as coding, debugging, building applications, websites, AI models and deployment. General requests outside that scope (for example, unrelated language lessons, jokes or recipes) are politely refused.

### Setup (Windows / VS Code terminal)

Keep `vibe_dataset_examples.json` in the same folder as `app.py`.

```powershell
ollama pull qwen3:1.7b
ollama create vibe-coding-assistant -f .\Modelfile
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
python test_retrieval.py
python app.py
```

`test_retrieval.py` checks the retrieval logic and scope gate and does not require an Ollama generation call. The `ollama create` command should be rerun after changing the `Modelfile`, because it rebuilds the customised Ollama model with the updated configuration.

### Controls

| Key | Action |
|---|---|
| Enter | Send message |
| Ctrl+J | Insert a new line |
| Ctrl+N | Start a new chat |
| Ctrl+L | Clear the current chat |
| Ctrl+C | Quit |

### Customising the scope gate

The terminal application's `app.py` contains the following configuration points:

- `SCOPE_WORDS` / `SCOPE_PHRASES`: add or remove terms that determine whether requests are in scope. For example, `unity` or `flutter` can be added if those development topics should be accepted.
- `REFUSAL_MESSAGE`: edit the fixed message returned for out-of-scope requests.
- `GREETING_MESSAGE`: edit the canned greeting response.
- `STRONG_RETRIEVAL_SCORE`: adjust how strongly a request must match a dataset row before it is accepted. A lower threshold is more permissive.

The gate is keyword-based, not a semantic safety classifier. A valid coding request without known terms may be rejected, while an unrelated request containing a broad word such as `app` may pass. Keep the system prompt as a secondary safeguard, but do not assume a small model will perfectly resolve mixed-scope requests.

### Strict mode versus general assistant mode

Use the **strict terminal edition** when the demonstration must stay within vibe-coding topics. Use the **general conversational assistant** when normal conversation and broader coding questions should be supported. These behaviours are intentionally different; do not enable a strict refusal gate in the general assistant unless that is the intended product behaviour.

---

## 10. Running the assistant locally

### One-time setup

1. Install Ollama, Python and VS Code.
2. Download the model required by the chosen assistant profile.
3. Create the customised Ollama model using its matching `Modelfile`.
4. Install the Python requirements in a virtual environment.
5. Place `vibe_dataset_examples.json` next to the correct assistant `app.py`.
6. Run the retrieval tests before launching the application.

### Normal demo launch

For the browser-based Qwen3 1.7B assistant:

```powershell
ollama list
python app.py
```

Open `http://127.0.0.1:7860` in a browser.

For the strict terminal edition:

```powershell
python test_retrieval.py
python app.py
```

For the original scratch-trained SLM:

```powershell
ollama run vibe-coding-qwen3
```

### Offline demonstration check

The first model download and Python package installation require internet access. After setup is complete:

1. Disconnect Wi-Fi.
2. Start or confirm that Ollama is running.
3. Launch the local assistant from VS Code.
4. Open the local interface or terminal session.
5. Test a normal request, a Vibe Coding request and a follow-up question.
6. Confirm that no Hugging Face API, hosted inference endpoint or external API key is required at runtime.

An optional Windows environment setting for disabling Ollama cloud features is:

```powershell
setx OLLAMA_NO_CLOUD 1
```

After changing this environment variable, completely quit and restart Ollama for the setting to take effect.

---

## 11. Testing checklist

Use the following prompts to test the full conversational assistant:

| Test | Prompt | What it checks |
|---|---|---|
| 1 | `Hi! How are you? What can you help me with?` | Greeting and normal conversation |
| 2 | `Create a Python star pyramid and explain how it works.` | Understanding a simple coding request and producing code/explanation |
| 3 | `Create a coding assistant. Give me a practical implementation plan.` | Dataset retrieval and task planning |
| 4 | `My React application keeps re-rendering infinitely. How should I troubleshoot it?` | Technical explanation |
| 5 | `Let's develop a chatbot. First explain the stages, then we will work on the first stage.` | Task breakdown and conversation memory |

For the strict terminal edition, also verify:

- A greeting receives the canned greeting reply.
- A recognised Vibe Coding request passes the scope gate.
- An unrelated request receives the fixed refusal without calling Ollama.
- A refused request is not stored in chat history.
- A valid follow-up after an accepted coding request is handled correctly.
- `python test_retrieval.py` passes before the live Ollama demo.

For the original scratch-trained model, use the four dataset-aligned prompts in the Ollama section. Evaluate its generated content separately from the pretrained 1.7B assistant; outputs from the two models must not be mixed in the results table.

---

## 12. Publishing the project to GitHub

GitHub should contain the source code, notebooks, documentation, reports and lightweight prepared dataset export needed to reproduce the work. Large model artifacts and large dataset files should remain on Hugging Face or another appropriate artifact store rather than being committed directly as ordinary Git files.

### Add a `.gitignore`

A starting point:

```gitignore
# Python environments and caches
.venv/
__pycache__/
*.py[cod]
.pytest_cache/

# Notebook and runtime checkpoints
.ipynb_checkpoints/
*.pt
*.pth

# Large model artifacts
*.gguf
*.safetensors

# Local secrets
.env
.env.*
!.env.example

# Local dataset caches and temporary files
.cache/
huggingface/
*.tmp
```

Check the ignore file before committing. Do not ignore reports or small source files that are required to reproduce the project.

### Initialize and push the repository

Create an empty repository on GitHub, then run these commands from the project root:

```powershell
git init
git add .
git status
git commit -m "Initial Vibe Coding SLM project"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
git push -u origin main
```

Replace the placeholder URL with the actual repository URL.

### What should be in GitHub

- End-to-end training notebook.
- Professor's reference notebook, if redistribution is permitted; otherwise provide a note and link/reference rather than copying it.
- Dataset export/preparation script.
- Assistant application source code.
- Both relevant Modelfiles.
- Retrieval tests and scope-gate tests.
- Requirements file(s).
- Dataset/model cards or links to their Hugging Face pages.
- Training, audit and evaluation reports.
- This README and any local deployment guide.

Do not commit Hugging Face tokens, passwords, private account files, full checkpoints or large GGUF/safetensors files into ordinary Git history. Link to the Hugging Face repositories instead. If a large artifact genuinely needs to live in GitHub, use Git LFS and verify the repository's storage/bandwidth limits first.

---

## Troubleshooting

### `ollama` is not recognised

Install Ollama for Windows, reopen PowerShell and verify:

```powershell
ollama --version
```

### Ollama is not reachable at `127.0.0.1:11434`

Start Ollama from Windows and check:

```powershell
ollama list
```

The local service must be running before the Python application can generate responses.

### The model is not listed after creating it

Run the correct `ollama create` command from the folder containing the matching Modelfile. Confirm that the `FROM` line points to the correct local GGUF file or downloaded base model.

### `vibe_dataset_examples.json` cannot be found

Place the file beside the application or update the path in the application configuration. The export script should generate the JSON from the prepared dataset in Drive/Colab.

### `NameError: name 're' is not defined`

If the application calls Python regular-expression functions, add this import near the top of `app.py`:

```python
import re
```

### Python syntax error in the Ollama request payload

Check that the request `payload` dictionary is closed before it is passed to `json.dumps`. This was a known issue in an earlier Windows demo version.

### GGUF conversion reports a tokenizer error

Check that the exported model directory contains the required Qwen3 tokenizer files and that the tokenizer configuration is compatible with the installed conversion tools. Validate the tokenizer before conversion and use the repaired tokenizer consistently in the Hugging Face export and GGUF build.

### The scratch-trained model loads but produces incoherent answers

A successful Ollama import confirms that the GGUF can be loaded; it does not confirm that the model has learned useful language behaviour. The original model was randomly initialized and trained on only 14 unique mappings. Its training diagnostics show overfitting and a 0% exact match on one held-out example. Do not try to fix this solely by increasing temperature or displaying a more elaborate interface. Use the separate pretrained Qwen3 1.7B assistant for the broader conversational demo, while retaining the 173.4M model as the scratch-training experiment.

### Thinking consumes the answer budget

For the hybrid assistant, thinking and final content must be handled separately. Do not render the internal thinking field as the final answer. Keep ordinary requests fast, enable thinking only for complex tasks, and reserve enough generation budget for the final response.

---

## Limitations and responsible reporting

1. **Only 14 unique mappings:** the source has approximately 1.1 million rows, but the audited normalized content contains only 14 distinct instruction/response mappings. The raw row count must not be presented as 1.1 million independent training tasks.
2. **Tiny held-out set:** the latest prepared split has one validation example and one test example. Metrics from one example are not sufficient to support broad generalization claims.
3. **Overfitting in the scratch-trained model:** training loss approached zero while validation loss increased. The latest diagnostic reported 0% exact match on its one held-out example.
4. **Pretrained assistant is a different component:** the general conversational experience comes from pretrained Qwen3 1.7B. Dataset retrieval grounds its Vibe Coding responses but does not mean the 1.7B model was trained from scratch on this dataset.
5. **Retrieval is not fine-tuning:** the assistant inserts retrieved examples into the prompt at runtime. This does not update the model's weights.
6. **No autonomous coding tools:** the documented assistant can explain, plan and generate text/code. It does not currently edit project files, run commands or execute generated code.
7. **Local runtime still needs validation:** the final Windows Ollama runtime, offline behaviour and response quality must be tested on the target laptop before claiming they are verified.

The academic report should clearly identify which model produced each result, which checkpoint was evaluated, which dataset split was used, and whether an answer was retrieved directly or generated by the local language model.

---

## References

- [Original assigned dataset — CodeDevX/Vibe-Coding-Instruct](https://huggingface.co/datasets/CodeDevX/Vibe-Coding-Instruct)
- [Qwen3-8B tokenizer on Hugging Face](https://huggingface.co/Qwen/Qwen3-8B)
- [Hugging Face Datasets: loading datasets](https://huggingface.co/docs/datasets/loading)
- [Hugging Face Hub: uploading files and folders](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Ollama documentation](https://docs.ollama.com/)
- [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama Modelfile documentation](https://docs.ollama.com/modelfile)

---

## Acknowledgement

This project uses the Vibe Coding dataset assigned for the academic project and the Qwen3 architecture/tokenizer ecosystem. Dataset ownership, licensing and redistribution conditions remain governed by the original dataset and model terms. Check the relevant model and dataset cards before redistributing their contents.
