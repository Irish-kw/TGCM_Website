# Figure 4 — blind unknown-K comparison

Run `reproduce.ipynb`. It prepares the synthetic validation, TGCM, neural-baseline, DECOMPOSE, and proprietary-LLM assets. The full run evaluates K=2 through K=6, seven TGCM embeddings, five checkpoint seeds, DANet, MossFormer2, DECOMPOSE, and both released LLM response sets.

DECOMPOSE uses an isolated CPU-only legacy environment because its original backend requires Python 3.7 and TensorFlow 1.15.5:

```bash
conda env create -f environment_decompose.yml
```

The notebook contains the original LLM prompt and optional live API examples. Released response records can be evaluated without making live API calls.