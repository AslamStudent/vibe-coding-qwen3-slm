# Deployment

This folder documents the optional GGUF/Ollama path. It is not considered complete until conversion succeeds and a local chat test is captured.

1. Run the optional GGUF conversion cell in the Colab notebook.
2. Verify the `.gguf` file exists and is valid.
3. If quantising, run quantisation only after successful F16 conversion.
4. Put the GGUF and a matching `Modelfile` on the computer running Ollama.
5. Check that `FROM` in the Modelfile points to the actual GGUF file.
6. Run `ollama create vibe-qwen3 -f Modelfile`.
7. Run `ollama run vibe-qwen3` and save screenshots of the real result.

Never rename `.pt` or `.safetensors` weights to `.gguf`.
