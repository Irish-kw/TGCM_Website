# TGCM Reproduction Artifact

This directory contains the public reproduction materials for **TGCM: Topic-Guided Consistency Modeling for One-Step Disentanglement of Interleaved APT Technique Sequences**.

## Environment

Linux, an NVIDIA GPU with CUDA 11.8 or newer, and approximately 16 GB of free disk space are recommended. CPU execution is supported for the TGCM, DANet, and MossFormer2 paths but is slower.

```bash
conda env create -f environment.yml
conda env create -f 01_Figure04_Blind_Unknown_K/environment_decompose.yml
conda activate tgcm-review
python -m ipykernel install --user --name tgcm-review --display-name "TGCM reviewer"
jupyter lab
```

Each notebook locates the repository root from its own directory and can be run independently.

## Paper-to-artifact index

| Paper item | Directory | Reproduction target |
|---|---|---|
| Figure 4 | `01_Figure04_Blind_Unknown_K` | Blind unknown-K comparison across TGCM, DANet, MossFormer2, DECOMPOSE, OpenAI, and Gemini. |
| Tables IV and XII | `02_Tables04_12_ZeroShot_Coverage` | Zero-shot kill-chain coverage. |
| Tables V and XIII | `03_Tables05_13_ZeroShot_Performance` | Zero-shot TGCM and DANet checkpoint evaluation. |
| Tables VI and XXIII | `04_Tables06_23_DARPA_Robustness` | DARPA TC-E5 perturbation robustness. |
| Tables VII and XXI | `05_Tables07_21_CAPTURE` | CAPTure single-host and multi-host end-to-end evaluation. |
| Table XXII | `06_Table22_Unknown_K` | Known-K versus Up-to-6 inference sensitivity. |
| Table XXV | `07_Table25_Embedding_Sensitivity` | Topic-encoder embedding sensitivity. |

## Full run and smoke test

The notebooks expose `FULL_REPRODUCTION`. Set it to `True` for the complete paper run. Set it to `False` only to verify the execution path on a small subset; smoke-test output is not a reported paper result.

## Data and checkpoints

`data/manifest.json` records the named evaluation assets, archive sizes, SHA-256 values, and extraction requirements. Large CAPture CSV archives and model assets are distributed separately from the static website and will be linked from the Downloads page.