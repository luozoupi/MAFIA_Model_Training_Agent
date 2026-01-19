# CudaForge: MCP Multi-Agent Finetuning Framework

CudaForge is focused on multi-agent finetuning workflows powered by MCP-style tooling.
The core goal is to orchestrate a planner, coder, and feedback agent to design, implement,
and improve task-specific finetuning runs. The legacy CUDA optimization workflow now lives
under `legacy/`.

<img src="./pic/human_agents_v2.png">

## 🔧 Build Environment
```
conda env create -f environment.yml
```

Note: Some packages may fail to install automatically when creating the environment.
If that happens, please install the missing packages manually using conda install or pip install.
```
pip install torch
pip install openai
pip install pandas
pip install matplotlib
pip install datasets peft bitsandbytes trl
```

For Hugging Face datasets or model downloads, set `HF_TOKEN` if needed.
## 🧩 MCP Multi-Agent Finetuning Workflow

This repo includes a starter MCP-style multi-agent workflow (planner → coder → feedback)
for finetuning tasks. It scans a lightweight model catalog, drafts a finetuning plan,
generates a training script, runs it, and asks a feedback agent for the next iteration.

Example run:

```bash
python3 mcp_main.py "Finetune a lightweight chat model for customer support classification" \
  --server_type openai \
  --model_name o3-mini \
  --model_catalog mcp/model_catalog.json
```

### MCP Backend Notes

- **API-backed models:** set the relevant API key (e.g., `OPENAI_API_KEY`) and use `--server_type openai`.
- **vLLM:** start a local OpenAI-compatible vLLM server and pass `--server_type vllm --server_address <host> --server_port <port>`.

The MCP flow writes all prompts, replies, and run artifacts under `mcp_runs/<timestamp>/llm_io` for easy inspection.
For a deeper walkthrough, see [`docs/finetuning.md`](docs/finetuning.md).

## 📦 Repository Layout (Finetuning)

- `mcp_main.py`: MCP multi-agent entry point (planner → coder → feedback).
- `mcp/`: tool registry + MCP tools + sample model catalog.
- `prompts/`: agent prompt templates.
- `docs/finetuning.md`: detailed workflow guide.
- `legacy/`: archived CUDA optimization workflow.
- `examples/`: example finetuning workflows (including YOLOv8).

## 🔁 Legacy CUDA Optimization

The original CUDA kernel optimization workflow still exists in the repository,
but it is no longer the primary focus. If you need it, refer to the historical
docs and scripts under `legacy/main.py` and related files.
