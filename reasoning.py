"""
reasoning.py — Clinical reasoning and risk assessment module

Takes structured extraction data and uses the NVIDIA Nemotron LLM
(via OpenAI-compatible API) to produce:
- Risk flag (Routine / Needs Review / Urgent) with justification
- Recommended next step for clinical/administrative staff
- Multi-document comparison when multiple documents are available

All LLM calls are timed and logged for performance profiling.
"""

import json
import logging
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

from extraction import ClinicalExtraction

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Pydantic Schema — Risk Assessment & Recommendations
# ─────────────────────────────────────────────────────────────

class RiskAssessment(BaseModel):
    """Risk assessment for a single clinical document."""
    risk_level: Literal["Routine", "Needs Review", "Urgent"] = Field(
        description="Triage risk level"
    )
    justification: str = Field(
        description="One-sentence justification for the risk level"
    )
    driving_factors: list[str] = Field(
        description="Specific extracted fields/values that drove the risk flag"
    )
    recommended_next_step: str = Field(
        description="Actionable recommendation for clinical/administrative staff"
    )


class MultiDocComparison(BaseModel):
    """Comparison across multiple documents for the same or related patients."""
    trending_observations: list[str] = Field(
        description="Trends identified across documents (e.g., worsening labs)"
    )
    conflicts: list[str] = Field(
        description="Conflicting information found across documents"
    )
    overall_risk_level: Literal["Routine", "Needs Review", "Urgent"] = Field(
        description="Overall risk considering all documents"
    )
    summary: str = Field(
        description="Brief narrative summary of the multi-document analysis"
    )


# ─────────────────────────────────────────────────────────────
# Risk Assessment Prompt
# ─────────────────────────────────────────────────────────────

RISK_ASSESSMENT_SYSTEM_PROMPT = """You are a clinical decision-support triage assistant. Given structured clinical data extracted from a medical document, you must assess the patient's risk level and recommend a next step.

You MUST return ONLY valid JSON matching this schema:

```json
{
  "risk_level": "Routine | Needs Review | Urgent",
  "justification": "One sentence explaining why this risk level was assigned",
  "driving_factors": ["List of specific clinical findings that drove the risk flag"],
  "recommended_next_step": "Actionable recommendation for staff"
}
```

## Risk Level Guidelines
- **Routine**: Normal findings, stable chronic conditions well-managed, routine follow-up. Example: annual wellness exam with all normal results.
- **Needs Review**: Suboptimal control of chronic conditions, new findings requiring follow-up, medication changes, borderline values. Example: elevated HbA1c, blood pressure above goal.
- **Urgent**: Critical lab values, acute decompensation, findings requiring immediate intervention. Example: critically elevated potassium, acute organ failure.

## Rules
1. ALWAYS cite which specific extracted values drove your risk flag (e.g., "Potassium 6.8 mEq/L", "eGFR 19 mL/min").
2. The recommended_next_step must be specific and actionable (not generic advice).
3. Consider the totality of findings — a single abnormal value in an otherwise healthy patient may be "Needs Review," while multiple critical values are "Urgent."
4. Return ONLY the JSON object. No markdown fences, no explanation."""


MULTI_DOC_SYSTEM_PROMPT = """You are a clinical decision-support system analyzing MULTIPLE clinical documents. Compare the extracted data across documents and identify trends, conflicts, and overall risk.

You MUST return ONLY valid JSON matching this schema:

```json
{
  "trending_observations": ["List of trends across documents"],
  "conflicts": ["List of conflicting information found"],
  "overall_risk_level": "Routine | Needs Review | Urgent",
  "summary": "Brief narrative summary of the multi-document analysis"
}
```

## Rules
1. Look for trends: worsening or improving lab values, changing medications, evolving symptoms.
2. Flag conflicts: different medication lists, inconsistent diagnoses, contradictory findings.
3. Note if documents are for the SAME patient or DIFFERENT patients.
4. The overall_risk_level should reflect the HIGHEST risk across all documents unless context suggests otherwise.
5. Return ONLY the JSON object. No markdown fences, no explanation."""


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


def _parse_json_response(raw_text: str) -> dict:
    """Clean and parse a JSON response from the LLM."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def assess_risk(
    extraction: ClinicalExtraction,
    openai_client,
    model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
    max_retries: int = 2,
) -> Optional[RiskAssessment]:
    """
    Assess the risk level of a clinical extraction.

    Parameters
    ----------
    extraction : ClinicalExtraction
        The structured extraction from a clinical document.
    openai_client : openai.OpenAI
        An initialized OpenAI client pointed at NVIDIA API.
    model : str
        The model to use.
    max_retries : int
        Maximum retry attempts.

    Returns
    -------
    RiskAssessment or None
        The validated risk assessment, or None if all attempts fail.
    """
    t0_total = time.perf_counter()
    logger.info(
        f"[TIMING] assess_risk: START — patient={extraction.patient_name.value}, model={model}"
    )

    extraction_json = extraction.model_dump_json(indent=2)

    for attempt in range(max_retries + 1):
        t0_attempt = time.perf_counter()
        try:
            logger.info(f"[TIMING] assess_risk: attempt {attempt + 1}/{max_retries + 1} — sending API request...")

            t0_api = time.perf_counter()
            stream = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RISK_ASSESSMENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Assess the risk level for the following clinical extraction:\n\n"
                            f"```json\n{extraction_json}\n```"
                        ),
                    },
                ],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 4096,
                },
                stream=True,
            )
            api_setup_elapsed = time.perf_counter() - t0_api
            logger.info(f"[TIMING] assess_risk: API stream created in {api_setup_elapsed:.3f}s")

            raw_text = _collect_streaming_content(stream)

            t0_parse = time.perf_counter()
            data = _parse_json_response(raw_text)
            assessment = RiskAssessment.model_validate(data)
            parse_elapsed = time.perf_counter() - t0_parse

            attempt_elapsed = time.perf_counter() - t0_attempt
            total_elapsed = time.perf_counter() - t0_total

            logger.info(f"[TIMING] assess_risk: JSON parse + validate: {parse_elapsed:.3f}s")
            logger.info(
                f"[TIMING] assess_risk: attempt {attempt + 1} succeeded in {attempt_elapsed:.3f}s — "
                f"risk_level={assessment.risk_level}"
            )
            logger.info(f"[TIMING] assess_risk: TOTAL: {total_elapsed:.3f}s")
            return assessment

        except json.JSONDecodeError as e:
            attempt_elapsed = time.perf_counter() - t0_attempt
            logger.warning(
                f"Attempt {attempt + 1}: JSON parse error in risk assessment after {attempt_elapsed:.3f}s: {e}"
            )
        except Exception as e:
            attempt_elapsed = time.perf_counter() - t0_attempt
            logger.warning(
                f"Attempt {attempt + 1}: Risk assessment error after {attempt_elapsed:.3f}s: {e}"
            )

    total_elapsed = time.perf_counter() - t0_total
    logger.error(f"All risk assessment attempts failed after {total_elapsed:.3f}s total.")
    return None


def compare_documents(
    extractions: list[ClinicalExtraction],
    openai_client,
    model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
    max_retries: int = 2,
) -> Optional[MultiDocComparison]:
    """
    Compare multiple clinical extractions to identify trends and conflicts.

    Parameters
    ----------
    extractions : list[ClinicalExtraction]
        Multiple extractions to compare.
    openai_client : openai.OpenAI
        An initialized OpenAI client pointed at NVIDIA API.
    model : str
        The model to use.
    max_retries : int
        Maximum retry attempts.

    Returns
    -------
    MultiDocComparison or None
        The comparison result, or None if all attempts fail.
    """
    if len(extractions) < 2:
        return None

    t0_total = time.perf_counter()
    logger.info(
        f"[TIMING] compare_documents: START — {len(extractions)} documents, model={model}"
    )

    docs_json = []
    for i, ext in enumerate(extractions):
        docs_json.append({
            "document_index": i + 1,
            "document_type": ext.document_type.value,
            "document_date": ext.document_date.value,
            "patient_name": ext.patient_name.value,
            "extraction": json.loads(ext.model_dump_json()),
        })

    combined = json.dumps(docs_json, indent=2)

    for attempt in range(max_retries + 1):
        t0_attempt = time.perf_counter()
        try:
            logger.info(
                f"[TIMING] compare_documents: attempt {attempt + 1}/{max_retries + 1} — sending API request..."
            )

            t0_api = time.perf_counter()
            stream = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MULTI_DOC_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Compare the following clinical document extractions:\n\n"
                            f"```json\n{combined}\n```"
                        ),
                    },
                ],
                temperature=0.6,
                top_p=0.95,
                max_tokens=8192,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 4096,
                },
                stream=True,
            )
            api_setup_elapsed = time.perf_counter() - t0_api
            logger.info(f"[TIMING] compare_documents: API stream created in {api_setup_elapsed:.3f}s")

            raw_text = _collect_streaming_content(stream)

            t0_parse = time.perf_counter()
            data = _parse_json_response(raw_text)
            comparison = MultiDocComparison.model_validate(data)
            parse_elapsed = time.perf_counter() - t0_parse

            attempt_elapsed = time.perf_counter() - t0_attempt
            total_elapsed = time.perf_counter() - t0_total

            logger.info(f"[TIMING] compare_documents: JSON parse + validate: {parse_elapsed:.3f}s")
            logger.info(
                f"[TIMING] compare_documents: attempt {attempt + 1} succeeded in {attempt_elapsed:.3f}s — "
                f"overall_risk={comparison.overall_risk_level}"
            )
            logger.info(f"[TIMING] compare_documents: TOTAL: {total_elapsed:.3f}s")
            return comparison

        except json.JSONDecodeError as e:
            attempt_elapsed = time.perf_counter() - t0_attempt
            logger.warning(
                f"Attempt {attempt + 1}: JSON parse error in comparison after {attempt_elapsed:.3f}s: {e}"
            )
        except Exception as e:
            attempt_elapsed = time.perf_counter() - t0_attempt
            logger.warning(
                f"Attempt {attempt + 1}: Multi-doc comparison error after {attempt_elapsed:.3f}s: {e}"
            )

    total_elapsed = time.perf_counter() - t0_total
    logger.error(f"All multi-doc comparison attempts failed after {total_elapsed:.3f}s total.")
    return None
