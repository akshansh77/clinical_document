"""
extraction.py — Clinical data extraction module

Defines the Pydantic schema for structured clinical extraction and uses
the NVIDIA Nemotron LLM (via OpenAI-compatible API) to extract structured
data from raw document text.
Every extracted field includes a confidence score and evidence snippet
for clinical traceability.

All LLM calls are timed and logged for performance profiling.
"""

import json
import logging
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Pydantic Schema — Structured Clinical Extraction
# ─────────────────────────────────────────────────────────────

class ExtractedField(BaseModel):
    """A single extracted value with confidence and provenance."""
    value: str = Field(description="The extracted value as a string")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    evidence_snippet: str = Field(
        description="The exact phrase from the source document that supports this extraction"
    )


class Medication(BaseModel):
    """A medication with name, dosage, and frequency."""
    name: ExtractedField
    dosage: ExtractedField
    frequency: ExtractedField


class Diagnosis(BaseModel):
    """A diagnosis with ICD-adjacent plain-language description."""
    description: ExtractedField


class VitalSign(BaseModel):
    """A vital sign or lab value with normal/abnormal status."""
    name: ExtractedField
    value: ExtractedField
    unit: ExtractedField
    status: ExtractedField  # "normal" or "abnormal"


class ClinicalExtraction(BaseModel):
    """Complete structured extraction from a clinical document."""
    patient_name: ExtractedField
    patient_age: ExtractedField
    patient_sex: ExtractedField
    chief_complaint: ExtractedField
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[ExtractedField] = Field(default_factory=list)
    vital_signs: list[VitalSign] = Field(default_factory=list)
    follow_up_instructions: list[ExtractedField] = Field(default_factory=list)
    document_type: ExtractedField
    document_date: ExtractedField


# ─────────────────────────────────────────────────────────────
# Extraction Prompt
# ─────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a clinical document analysis system. Your task is to extract structured information from medical documents.

You MUST return ONLY valid JSON matching the schema below. Do not include any text outside the JSON.

## Schema

```json
{
  "patient_name": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
  "patient_age": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
  "patient_sex": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
  "chief_complaint": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
  "diagnoses": [
    {"description": {"value": "ICD-adjacent plain language description", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"}}
  ],
  "medications": [
    {
      "name": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
      "dosage": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
      "frequency": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"}
    }
  ],
  "allergies": [
    {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"}
  ],
  "vital_signs": [
    {
      "name": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
      "value": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
      "unit": {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
      "status": {"value": "normal|abnormal", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"}
    }
  ],
  "follow_up_instructions": [
    {"value": "string", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"}
  ],
  "document_type": {"value": "string (e.g., Discharge Summary, Lab Report, Intake Form, Physician Note)", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"},
  "document_date": {"value": "MM/DD/YYYY or as stated", "confidence": 0.0-1.0, "evidence_snippet": "exact quote"}
}
```

## Rules
1. For `evidence_snippet`, use the EXACT text from the document (a short phrase, not the whole document).
2. Set `confidence` based on how clearly the information is stated:
   - 0.95-1.0: explicitly and unambiguously stated
   - 0.80-0.94: clearly stated but requires minor interpretation
   - 0.60-0.79: inferred from context with reasonable certainty
   - 0.40-0.59: weakly supported, may be ambiguous
   - Below 0.40: very uncertain, consider omitting
3. For `vital_signs.status`, use "normal" or "abnormal" based on standard reference ranges.
4. Include ALL medications, diagnoses, vital signs, and lab values mentioned in the document.
5. If a field is not present in the document, use value "Not documented", confidence 0.0, and evidence_snippet "N/A".
6. For lab reports, include all lab values as vital_signs entries with their flag status.
7. Return ONLY the JSON object. No markdown fences, no explanation."""


def _collect_streaming_content(stream) -> str:
    """Collect the full content from a streaming OpenAI response, skipping reasoning tokens."""
    content_parts = []
    token_count = 0
    t0 = time.perf_counter()
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content is not None:
            content_parts.append(delta.content)
            token_count += 1
    elapsed = time.perf_counter() - t0
    logger.info(
        f"[TIMING] _collect_streaming_content: {elapsed:.3f}s — "
        f"{token_count} content chunks, {sum(len(p) for p in content_parts)} chars total"
    )
    return "".join(content_parts)


def extract_clinical_data(
    document_text: str,
    openai_client,
    model: str = "nvidia/nemotron-3-ultra-550b-a55b",
    max_retries: int = 2,
) -> Optional[ClinicalExtraction]:
    """
    Extract structured clinical data from document text using NVIDIA Nemotron LLM.

    Parameters
    ----------
    document_text : str
        The raw text of the clinical document.
    openai_client : openai.OpenAI
        An initialized OpenAI client pointed at NVIDIA API.
    model : str
        The model to use.
    max_retries : int
        Maximum number of retry attempts for malformed responses.

    Returns
    -------
    ClinicalExtraction or None
        The validated extraction, or None if all attempts fail.
    """
    t0_total = time.perf_counter()
    logger.info(
        f"[TIMING] extract_clinical_data: START — "
        f"input text length={len(document_text)} chars, model={model}"
    )

    for attempt in range(max_retries + 1):
        t0_attempt = time.perf_counter()
        try:
            logger.info(f"[TIMING] extract_clinical_data: attempt {attempt + 1}/{max_retries + 1} — sending API request...")

            t0_api = time.perf_counter()
            stream = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Extract all clinical information from the following document:\n\n---\n{document_text}\n---",
                    },
                ],
                temperature=0.6,
                top_p=0.95,
                max_tokens=16384,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 8192,
                },
                stream=True,
            )
            api_setup_elapsed = time.perf_counter() - t0_api
            logger.info(f"[TIMING] extract_clinical_data: API stream created in {api_setup_elapsed:.3f}s")

            # Collect the full content (skipping reasoning tokens)
            raw_text = _collect_streaming_content(stream).strip()

            # Clean up potential markdown code fences
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                # Remove first line (```json) and last line (```)
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw_text = "\n".join(lines).strip()

            # Parse and validate
            t0_parse = time.perf_counter()
            data = json.loads(raw_text)
            extraction = ClinicalExtraction.model_validate(data)
            parse_elapsed = time.perf_counter() - t0_parse

            attempt_elapsed = time.perf_counter() - t0_attempt
            total_elapsed = time.perf_counter() - t0_total

            logger.info(
                f"[TIMING] extract_clinical_data: JSON parse + validate: {parse_elapsed:.3f}s"
            )
            logger.info(
                f"[TIMING] extract_clinical_data: attempt {attempt + 1} succeeded in {attempt_elapsed:.3f}s — "
                f"{len(extraction.diagnoses)} diagnoses, "
                f"{len(extraction.medications)} medications, "
                f"{len(extraction.vital_signs)} vital signs"
            )
            logger.info(
                f"[TIMING] extract_clinical_data: TOTAL: {total_elapsed:.3f}s"
            )
            return extraction

        except json.JSONDecodeError as e:
            attempt_elapsed = time.perf_counter() - t0_attempt
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1}: JSON parse error after {attempt_elapsed:.3f}s: {e}"
            )
        except Exception as e:
            attempt_elapsed = time.perf_counter() - t0_attempt
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1}: Extraction error after {attempt_elapsed:.3f}s: {e}"
            )

    total_elapsed = time.perf_counter() - t0_total
    logger.error(f"All extraction attempts failed after {total_elapsed:.3f}s total.")
    return None
