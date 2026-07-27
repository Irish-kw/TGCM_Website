"""Credential-free source for the Figure 4 proprietary-LLM calls.

Credentials are read only from environment variables or Application Default
Credentials. The prompt templates are literals so reviewers can inspect the
exact request sent to each provider.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


PROMPT_PATH = Path(__file__).resolve().parents[1] / "paper_metadata" / "figure04_llm_prompt.txt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(sequence: list[str], k_max: int = 6) -> str:
    # k_max is accepted for a common live-call interface. The released prompt
    # did not disclose true K or interpolate Kmax; only the sequence is filled.
    del k_max
    return PROMPT_TEMPLATE.format(question=json.dumps(sequence, ensure_ascii=False))


def call_openai(sequence: list[str], model: str = "gpt-5.5-extra-high", k_max: int = 6) -> dict:
    """Call OpenAI using ``OPENAI_API_KEY``; no key is stored in this artifact."""

    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before making a live OpenAI request.")
    response = OpenAI().chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": build_prompt(sequence, k_max)},
        ],
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return json.loads(text)


def call_gemini(
    sequence: list[str],
    model: str = "gemini-3-flash-preview",
    k_max: int = 6,
    project: str | None = None,
    location: str | None = None,
) -> dict:
    """Call Gemini on Vertex AI with ADC; no service-account key is stored."""

    from google import genai
    from google.genai import types

    project_id = project or os.getenv("GOOGLE_CLOUD_PROJECT")
    region = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    if not project_id:
        raise RuntimeError(
            "Set GOOGLE_CLOUD_PROJECT and configure Application Default Credentials."
        )
    client = genai.Client(vertexai=True, project=project_id, location=region)
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(sequence, k_max),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return json.loads(response.text)


def append_response_record(
    output: str | Path,
    *,
    sequence: list[str],
    true_y_z: list[int],
    response: dict,
) -> Path:
    """Append one result in the schema used by the released complete PKLs."""

    import pickle

    path = Path(output)
    records = []
    if path.is_file():
        with path.open("rb") as stream:
            records = pickle.load(stream)
    records.append(
        {
            "sequence": list(sequence),
            "true_y_z": list(map(int, true_y_z)),
            "llm_locations": list(map(int, response["locations"])),
        }
    )
    with path.open("wb") as stream:
        pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return path
