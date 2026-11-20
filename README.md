# Vibe Coding Qwen3 Small Language Model (SLM)

A compact, Qwen3-architecture causal language model built for a focused coding-instruction experiment using the **CodeDevX/Vibe-Coding-Instruct** dataset.

This repository documents the complete project workflow: understanding the dataset, auditing its quality, preparing training examples, defining and training a compact model, evaluating its outputs, publishing model/data artifacts to Hugging Face, and preparing a local GGUF/Ollama demonstration.

> **Important:** This is an experimental student project and proof of workflow. The current dataset audit found very limited response diversity despite the large row count. The model's evaluation results must therefore be interpreted as reproduction of recurring patterns, not proof of broad coding ability.

---

## Contents

- [1. Project at a glance](#1-project-at-a-glance)
- [2. Why this project exists](#2-why-this-project-exists)
- [3. Dataset](#3-dataset)
- [4. Dataset audit and limitations](#4-dataset-audit-and-limitations)
- [5. Model design](#5-model-design)
- [6. Training method](#6-training-method)
- [7. Repository structure](#7-repository-structure)
- [8. Requirements](#8-requirements)
- [9. Run the project in Google Colab](#9-run-the-project-in-google-colab)
- [10. Understand the notebook sections](#10-understand-the-notebook-sections)
- [11. Evaluation and results](#11-evaluation-and-results)
- [12. Hugging Face model and dataset](#12-hugging-face-model-and-dataset)
- [13. GGUF and Ollama deployment](#13-gguf-and-ollama-deployment)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Reproducibility checklist](#15-reproducibility-checklist)
- [16. Limitations and responsible reporting](#16-limitations-and-responsible-reporting)
- [17. References](#17-references)

---

## 1. Project at a glance

| Item | Details |
|---|---|
| Project | Vibe Coding Qwen3 SLM |
| Task | Focused coding-instruction response generation |
| Source dataset | `CodeDevX/Vibe-Coding-Instruct` |
| Model class | Hugging Face `Qwen3ForCausalLM` |
| Initialisation | Randomly initialised model weights |
| Tokenizer | `Qwen/Qwen3-8B` tokenizer |
| Approximate parameter count | 173.4 million (confirm exact count in your run log) |
| Transformer layers | 6 |
| Hidden size | 512 |
| Attention heads | 8 |
| Key/value heads | 2 |
| Intermediate size | 1,536 |
| Maximum sequence length | 128 tokens |
| Training update ceiling | 1,200 optimizer updates |
| Training framework | PyTorch + Hugging Face Transformers |
| Deployment target | GGUF + Ollama (only after successful conversion and local testing) |

### One important distinction

The project uses the **Qwen3 architecture**, but it does not load the pretrained weights of the 8-billion-parameter Qwen3 model. It creates a new compact model configuration with randomly initialised weights and trains that model on the project dataset.

The Qwen3-8B name refers to the **tokenizer source**, not the size or pretrained knowledge of the trained model. A tokenizer converts text into token IDs; it does not provide pretrained language-model weights.

---

## 2. Why this project exists

Large general-purpose language models can require substantial compute and memory. This project explores a smaller, task-focused model and a workflow that can be studied, reproduced and potentially run locally.

The project follows a Design Thinking process:

1. **Empathise:** identify the coding-support tasks and difficulties experienced by the intended users. Any interview or survey claims should be included only if real user research was conducted.
2. **Define:** frame the need for a focused coding-instruction assistant.
3. **Ideate:** consider the dataset, compact architecture, training workflow and local deployment.
4. **Prototype:** audit and prepare the data, train the model and connect a local demo.
5. **Test:** inspect generated answers, compare them with expected outputs, document failures and improve the system.

The purpose is not to claim that a 173-million-parameter model replaces a large coding model. It is to demonstrate and evaluate an end-to-end SLM development workflow.

---

## 3. Dataset

The project uses the dataset:

**CodeDevX / Vibe-Coding-Instruct**

Dataset page: <https://huggingface.co/datasets/CodeDevX/Vibe-Coding-Instruct>

The notebook loads it directly using the Hugging Face Datasets library:

```python
from datasets import load_dataset

dataset = load_dataset("CodeDevX/Vibe-Coding-Instruct")
print(dataset)
```

The project expects fields including:

- `instruction`: the coding-related task or request.
- `input`: optional additional context.
- `output`: the expected response.

The notebook checks that `instruction` and `output` are present before proceeding. It also prints the dataset splits, column names and an example row.

### Dataset licensing

The dataset page identifies its dataset licence. Review the current licence and its conditions before redistributing the dataset or derived files. This repository links to the original dataset rather than assuming that every source-data file should be copied into GitHub.

---

## 4. Dataset audit and limitations

A central finding of this project is that **a large number of rows does not automatically mean a large number of different examples**.

The audit in the notebook:

- counts records;
- checks blank instructions and blank outputs;
- normalises scenario-number variations in instructions;
- counts unique normalised instruction templates;
- counts unique output strings;
- checks whether one template is associated with multiple outputs;
- identifies scenario IDs.

The project audit reported approximately 1.1 million rows but only 14 normalised instruction templates and 14 fixed output strings. Re-run the audit and record the exact numbers from your current dataset version before publishing final results.

### Why repeated data matters

If many records contain the same instruction and same response, the model can learn to reproduce those repeated mappings. It may achieve a low training loss without learning how to solve genuinely new coding problems.

The notebook groups rows by scenario ID to prevent the same scenario ID appearing in multiple splits. However, the same normalised instruction templates and fixed outputs can still appear in train, validation and test under different scenario IDs.

Therefore, the split is **scenario-grouped**, not a fully independent template-held-out test.

---

## 5. Model design

The model is created with the Hugging Face Qwen3 implementation:

```python
from transformers import Qwen3Config, Qwen3ForCausalLM

config = Qwen3Config(
    vocab_size=len(tokenizer),
    hidden_size=512,
    intermediate_size=1536,
    num_hidden_layers=6,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=128,
)

model = Qwen3ForCausalLM(config)
```

The actual notebook sets additional configuration values, including the vocabulary and special-token IDs, RoPE settings, RMSNorm epsilon, attention dropout and embedding-tying behaviour. Use the notebook as the source of truth for the complete configuration.

### What the components do

- **Tokenizer:** converts words and symbols into token IDs.
- **Embedding layer:** converts token IDs into vectors.
- **Transformer layers:** process context using causal self-attention and feed-forward transformations.
- **Causal attention:** allows each token to use earlier context when predicting the next token.
- **LM head:** converts the final token representations into next-token scores.
- **Generation:** repeatedly predicts the next token until the output ends or the generation limit is reached.

---

## 6. Training method

The notebook uses a response-only training objective.

Each example is formatted in this general structure:

```text
### Instruction:
Create a coding assistant.

### Response:
The expected answer goes here.
```

The prompt tokens are assigned the label `-100`, which tells the model's loss function to ignore those positions. The response tokens and end-of-sequence token contribute to the training loss.

### Main training settings

| Setting | Notebook configuration |
|---|---:|
| Random seed | 42 |
| Maximum sequence length | 128 |
| Micro-batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Maximum optimizer updates | 1,200 |
| Learning rate | 0.0002 |
| Minimum learning rate | 0.00002 |
| Warm-up updates | 60 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Validation interval | Every 100 updates |
| Optimizer | AdamW |
| Learning-rate schedule | Warm-up + cosine decay |
| Precision | BF16 when supported, otherwise FP16 |

The reference notebook's 150,000 iterations are treated as an upper ceiling, not as the number of updates completed in this project run. The practical run cap is 1,200 optimizer updates, subject to the time limit and early stopping.

The notebook saves checkpoints and training history to Google Drive so that a Colab disconnection does not automatically destroy the saved progress.

---

## 7. Repository structure

The recommended first GitHub version is deliberately simple and matches the actual notebook-based project:

```text
vibe-coding-qwen3-slm/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── notebooks/
│   ├── Vibe_Coding_Qwen3_SLM_Final.ipynb
│   └── reference_notebook/              # optional, if permitted to share
│
├── docs/
│   ├── project_overview.md
│   ├── design_thinking.md
│   ├── dataset_card.md
│   ├── model_card.md
│   └── experiments.md
│
├── results/
│   ├── training_history.json
│   ├── evaluation_results.json
│   └── figures/
│       ├── training_loss.png
│       └── validation_loss.png
│
├── deployment/
│   ├── README.md
│   └── Modelfile                  # only if GGUF conversion succeeded
│
└── assets/
    ├── architecture.png
    └── screenshots/
```

Do not create empty Python modules that imply the project has been refactored into a package if the actual implementation still lives in the notebook. A later improvement can move stable functions into `src/`, but the repository should accurately represent the code that works today.

### Files that should not normally be committed

Do not push large or private runtime artefacts into ordinary Git history:

- full raw dataset downloads;
- the complete processed dataset cache;
- model checkpoint files such as `best.pt` and `latest.pt`;
- large `.safetensors` model weights;
- large `.gguf` files;
- Google Drive credentials or Hugging Face access tokens;
- local virtual environments and cache folders.

Host model weights and prepared datasets on Hugging Face, then link to them from this README.

---

## 8. Requirements

The notebook installs these main packages in its Colab environment:

```bash
pip install -U "transformers>=4.51.0" datasets accelerate huggingface_hub safetensors sentencepiece tqdm matplotlib
```

It also uses:

- Python
- PyTorch
- NumPy
- Google Colab's Drive integration
- CUDA-enabled GPU runtime

The exact PyTorch and CUDA versions depend on the Colab runtime. Record the versions printed by your actual run in `docs/experiments.md`.

A starter `requirements.txt` is included in this repository. For a strict reproduction, pin the exact versions from a successful run rather than relying indefinitely on unpinned latest versions.

---

## 9. Run the project in Google Colab

The primary training workflow is currently notebook-based. The simplest way for a new user to reproduce it is Google Colab.

### Step 1 — Open the notebook

1. Open Google Colab: <https://colab.research.google.com/>
2. Select **File → Upload notebook**.
3. Upload `notebooks/Vibe_Coding_Qwen3_SLM_Final.ipynb` from this repository.
4. Wait for the notebook to finish loading.

### Step 2 — Select a GPU

1. Open **Runtime → Change runtime type**.
2. Select a GPU runtime, such as a T4 when available.
3. Save the setting.
4. Run the runtime-check cell.

The notebook checks whether CUDA is available and prints the GPU name and memory. It stops if a GPU is unavailable.

### Step 3 — Mount Google Drive

When prompted, authorise Colab to access your Google Drive. The notebook creates a project folder containing:

- `checkpoints/`
- `hf_model/`
- `prepared_dataset/`
- training history and evaluation files.

Keep the Drive folder intact if you want to resume an interrupted run.

### Step 4 — Run the notebook from top to bottom

Use **Runtime → Run all**, or execute each cell in order.

Do not skip the data audit, split, tokenisation or model-configuration sections. They define what the training run actually means.

### Step 5 — Monitor training

During training, observe:

- optimizer update count;
- training loss;
- validation loss;
- elapsed time;
- checkpoint-saving messages.

The best checkpoint is selected using validation loss. The notebook also saves a latest checkpoint for resuming.

### Step 6 — Generate a sample

After training, the notebook loads the best checkpoint and generates a response to a sample coding instruction.

Inspect the generated text. A completed training loop does not guarantee that the generated answer is correct or useful.

### Step 7 — Evaluate

The evaluation cell compares generated responses with expected responses using:

- exact match after normalising whitespace and letter case;
- normalised text similarity using `SequenceMatcher`.

These are simple text-comparison metrics. They do not prove that a coding plan is technically correct.

### Step 8 — Save and publish artefacts

The notebook saves a Hugging Face-compatible model folder and a prepared dataset with train, validation and test splits. The upload cells create separate Hugging Face repositories for the model and dataset.

Use a Hugging Face write token through the notebook's login flow. Never paste a token into a source file or commit it to GitHub.

---

## 10. Understand the notebook sections

| Notebook section | What it does |
|---|---|
| Install packages | Installs the Python libraries |
| Runtime and Drive | Checks GPU and creates persistent folders |
| Reproducibility | Sets random seeds and precision settings |
| Load dataset | Downloads the required Hugging Face dataset |
| Dataset audit | Counts duplicates, templates, blanks and outputs |
| Scenario split | Creates train/validation/test groups |
| Tokenizer and formatting | Builds instruction/response text |
| Tokenisation | Converts text to IDs and masks prompt labels |
| Model definition | Creates a compact Qwen3 model from random weights |
| Training configuration | Sets batch size, learning rate and update limits |
| Data loaders | Batches and pads tokenised examples |
| Training loop | Optimises the model and saves checkpoints |
| Resume | Optionally restores a previous checkpoint |
| Generation | Produces a response from the best checkpoint |
| Evaluation | Calculates exact match and text similarity |
| Model export | Saves model and tokenizer files |
| Dataset export | Saves prepared train/validation/test splits |
| Hugging Face upload | Publishes model and dataset repositories |
| GGUF conversion | Optionally converts the model for local inference |
| Ollama instructions | Imports the GGUF file and runs a local chat |
| Final report | Prints the run summary and completion checks |

---

## 11. Evaluation and results

Add the exact results from your final successful run to `results/evaluation_results.json` and summarise them here.

**Do not report numbers from an older run as if they came from the latest run.** The notebook writes the evaluation summary, including the number of unique test prompts, exact matches, exact-match rate, mean normalised similarity, best checkpoint step and best validation loss.

The dataset's repeated patterns make the evaluation narrow. Even a high exact-match score can mean that the model reproduced known answers for recurring templates. It does not establish performance on unseen coding problems.

Include at least:

- the number of unique test prompts;
- exact matches and total test prompts;
- exact-match rate;
- mean normalised similarity;
- best checkpoint step and validation loss;
- at least two successful examples;
- at least two failed or incomplete examples;
- a short explanation of why each failure occurred.

---

## 12. Hugging Face model and dataset

The project publishes two separate repositories:

1. **Model repository:** contains the model configuration, weights, tokenizer and model card.
2. **Dataset repository:** contains the prepared train, validation and test splits.

Add the verified URLs here after checking that both repositories are public and their files can be opened:

- Model: `https://huggingface.co/REPLACE_WITH_YOUR_USERNAME/Vibe-Coding-Qwen3-SLM-Final`
- Prepared dataset: `https://huggingface.co/datasets/REPLACE_WITH_YOUR_USERNAME/Vibe-Coding-Instruct-Prepared`

If the repositories were created under different names in an earlier run, use the actual repository URLs instead of these example paths.

### What the model repository should explain

- architecture and parameter count;
- random initialisation (no pretrained Qwen3 weights);
- tokenizer source;
- training dataset;
- training configuration;
- evaluation results;
- known limitations;
- loading instructions.

---

## 13. GGUF and Ollama deployment

GGUF conversion is optional in the notebook. It is not complete merely because the code cell exists.

The intended deployment flow is:

```text
Hugging Face model folder
        |
        v
llama.cpp conversion
        |
        v
F16 GGUF file
        |
        v
Quantisation (if selected)
        |
        v
GGUF model + Modelfile
        |
        v
Ollama
        |
        v
Local chat test
```

### Step 1 — Confirm conversion succeeded

The notebook's GGUF conversion option is disabled by default. Enable it only when you are ready to attempt conversion.

A valid `.gguf` file must be created. Do not rename a `.pt` or `.safetensors` file and call it GGUF; these are different file formats.

### Step 2 — Quantise only after F16 conversion succeeds

The notebook includes an optional Q4_K_M quantisation step. Run it only after the F16 GGUF file exists and the quantisation tool has built successfully.

### Step 3 — Install Ollama

Install Ollama from its official website:

<https://ollama.com/>

### Step 4 — Create the Ollama model

Place the valid GGUF file and `Modelfile` in the same local folder. Confirm that the `FROM` path inside the Modelfile points to the actual GGUF file.

Then run:

```bash
ollama create vibe-qwen3 -f Modelfile
```

### Step 5 — Start a chat

```bash
ollama run vibe-qwen3
```

Enter a test coding instruction and inspect the response.

### Step 6 — Record the result

Add screenshots showing:

- the Ollama model creation completing successfully;
- the model appearing in `ollama list`;
- a real prompt entered by the user;
- the response generated by the local model.

If GGUF conversion or Ollama rejects the architecture, record the failure honestly and use the Hugging Face Transformers inference path until compatibility is resolved.

---

## 14. Troubleshooting

### CUDA is unavailable

- In Colab, select **Runtime → Change runtime type → GPU**.
- Re-run the runtime-check cell.
- If no GPU is available, wait for a GPU runtime or use another supported GPU environment.

### Colab disconnects during training

- Check that checkpoints are being saved to Google Drive.
- Reconnect the runtime.
- Run the setup cells again.
- Set `RESUME_FROM_LATEST = True` in the resume section and execute that cell after recreating the model, optimizer and scheduler.
- Confirm that the checkpoint path exists before resuming.

### Out-of-memory error

The current model and training settings were designed around a limited GPU budget, but available GPU memory varies. Do not randomly change many settings at once. Reduce the micro-batch size first, retain gradient accumulation if appropriate, and document every changed setting because it changes the experiment.

### Hugging Face upload fails

- Confirm that you are logged in with a write-enabled token.
- Confirm that the repository ID is correct.
- Do not expose the token in notebook output, screenshots or GitHub.
- Retry the upload after confirming the model and dataset folders exist.

### GGUF conversion fails

- Confirm that the model export folder contains the expected config, weights and tokenizer files.
- Update or rebuild the llama.cpp conversion environment as required by its current documentation.
- Read the complete error message.
- Do not claim Ollama support until a valid GGUF model has been imported and tested.

### Ollama cannot find the GGUF file

- Check the `FROM` path in the Modelfile.
- Use an absolute path if needed.
- Confirm the file exists and has a `.gguf` extension.
- Re-run `ollama create` after correcting the path.

---

## 15. Reproducibility checklist

Before marking the project as complete, verify each item:

- [ ] The original dataset page is linked.
- [ ] The notebook runs from a fresh Colab GPU runtime.
- [ ] The data audit prints exact row, template and output counts.
- [ ] The scenario-grouped split has no scenario-ID overlap.
- [ ] The notebook records the actual number of optimizer updates.
- [ ] The best checkpoint is saved.
- [ ] Training and validation history are saved.
- [ ] Evaluation JSON is saved.
- [ ] Model configuration and parameter count are recorded.
- [ ] Model repository is accessible on Hugging Face.
- [ ] Prepared dataset repository is accessible on Hugging Face.
- [ ] GGUF conversion is marked complete only if a valid file exists.
- [ ] Ollama is marked complete only after a real local chat test.
- [ ] Screenshots and experiment evidence are included.
- [ ] No access tokens, passwords, private keys or large checkpoint files are committed.

---

## 16. Limitations and responsible reporting

This project has several important limitations:

1. **Low data diversity:** approximately 1.1 million source rows reduce to a very small number of normalised instruction templates and fixed outputs in the current audit.
2. **Template overlap:** scenario grouping prevents scenario-ID overlap, but recurring templates and answers can still appear across splits.
3. **Random initialisation:** the model does not inherit the language understanding of pretrained Qwen3 weights.
4. **Short context:** the maximum sequence length is 128 tokens, limiting long instructions and answers.
5. **Small training budget:** the practical run is capped at 1,200 optimizer updates, not the 150,000-iteration reference ceiling.
6. **Narrow evaluation:** exact match and text similarity do not assess correctness, safety, maintainability or usefulness of generated code plans.
7. **Deployment compatibility:** GGUF conversion and Ollama support must be verified on the actual exported model.

The correct conclusion is that this is an end-to-end SLM prototype and a useful experiment in data auditing, model construction, training and deployment. Broader coding capability has not yet been established.

---

## 17. References

1. CodeDevX, **Vibe-Coding-Instruct** dataset: <https://huggingface.co/datasets/CodeDevX/Vibe-Coding-Instruct>
2. Qwen, **Qwen3-8B tokenizer/model page**: <https://huggingface.co/Qwen/Qwen3-8B>
3. Hugging Face, **Transformers documentation**: <https://huggingface.co/docs/transformers>
4. Hugging Face, **Datasets documentation**: <https://huggingface.co/docs/datasets>
5. Hugging Face, **Hub documentation**: <https://huggingface.co/docs/hub>
6. llama.cpp, **GGUF conversion and inference tools**: <https://github.com/ggml-org/llama.cpp>
7. Ollama, **Official documentation**: <https://docs.ollama.com/>

---

## Maintainer

**Project:** Vibe Coding Qwen3 SLM  
**Institution:** JAIN (Deemed-to-be University)  
**Programme:** B.Tech — Computer Science and Engineering (AI & ML)

For questions or issues, open a GitHub Issue in this repository.
