# Figure 4 — blind unknown-K comparison

The full Figure 4 notebook remains disabled while the multi-baseline release is finalized. A standalone reviewer-safe LLM notebook is available as `reproduce_llm.ipynb` for the OpenAI and Gemini call path.

The LLM notebook is independent of TGCM checkpoints and other notebooks. It downloads the exact public Figure 4 prompt, reads credentials only from environment variables / Application Default Credentials, contains no API keys, and stores no executed outputs in the distributed `.ipynb` file.

Environment for the LLM-only notebook:

```bash
pip install -U openai google-genai
```

Credentials:

- OpenAI: `OPENAI_API_KEY`
- Gemini API: `GEMINI_API_KEY`
- or Vertex AI: `GOOGLE_CLOUD_PROJECT` with Application Default Credentials

The paper labels the proprietary baselines as ChatGPT-5.5 and Gemini-3. Model identifiers remain environment-overridable in the notebook so reviewers can select the exact provider model/snapshot available to their account.

The complete Figure 4 run additionally evaluates K=2 through K=6, seven TGCM embeddings, five checkpoint seeds, DANet, MossFormer2, DECOMPOSE, and the released LLM response sets. DECOMPOSE uses an isolated CPU-only legacy environment because its original backend requires Python 3.7 and TensorFlow 1.15.5.
