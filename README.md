# 🏥 Clinical Document Intelligence

AI-powered clinical document extraction and risk assessment prototype. Upload unstructured clinical documents (PDFs, images, or text) and get structured, decision-ready patient summary cards with transparent risk flags and evidence-backed recommendations.

> ⚠️ **This prototype uses synthetic data only.** No real patient data is processed, stored, or transmitted. All sample documents are AI-generated for demonstration purposes.

---

## 🧠 The Approach

This system follows a three-stage pipeline to ensure clinical reasoning is applied transparently: **Extract → Reason → Present**.
1. **Extract:** Uploaded documents are parsed (using `pdfplumber` for PDFs, `pytesseract` for image OCR, or plain text passthrough). The raw text is fed to a large language model with a strict JSON schema prompt to extract patient demographics, diagnoses, medications, and vital signs. Every extracted data point includes a confidence score and a citation snippet traced back to the source text.
2. **Reason:** A secondary reasoning layer passes the structured extraction through a separate LLM call. This assesses the clinical data to assign a structured risk flag (Routine / Needs Review / Urgent), cite the specific clinical driving factors, and generate an actionable recommended next step.
3. **Present:** The Streamlit UI renders the results as premium, glassmorphic summary cards with color-coded risk badges, abnormal-value highlighting, and confidence indicators.

## 🛠️ AI Models & Tools Used

| Component | Tool | Justification |
|:---|:---|:---|
| **LLM Reasoning & Extraction** | NVIDIA Nemotron (`nemotron-3-ultra-550b-a55b`) | A 550B-parameter reasoning model accessed via the NVIDIA API. Chosen for its advanced reasoning capabilities required to accurately assess clinical risk and reliably output strict Pydantic JSON schemas. |
| **User Interface** | Streamlit | Allows for rapid prototyping of a single-page app with file upload capabilities and rich custom HTML/CSS rendering for premium aesthetics. |
| **Parsing Pipeline** | pdfplumber, pytesseract, pdf2image | Provides a robust fallback mechanism to handle various clinical document formats (digital PDFs, scanned images, and raw text). |
| **Data Validation** | Pydantic v2 | Enforces strict typing on the LLM output, automatically catching malformed JSON responses and ensuring the UI always receives valid structured data. |

## 📝 Assumptions Made

- **Synthetic Data:** All sample documents and scenarios are entirely AI-generated. No real PHI is used.
- **Language:** The extraction prompts and parsing assume English-language clinical documents.
- **Standalone Prototype:** The system does not integrate with EHR systems or external databases; all knowledge is derived zero-shot from the document context.
- **OCR Quality:** Tesseract accuracy varies with document quality; handwritten clinical notes may not extract perfectly.

## 💻 Setup Instructions

1. **Prerequisites:** Python 3.11+, and an NVIDIA API key (a demo key is pre-filled in the app). For OCR capabilities, [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) must be installed on your system.
2. **Installation:**
   ```bash
   git clone <repo-url>
   cd clinical-doc-intel
   python -m venv .venv
   # Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   *The app will be available at `http://localhost:8501/`*

## 🎨 Design Notes

The UI was designed to feel like a **state-of-the-art premium medical dashboard**. 
- **Aesthetic:** We utilized a Cyberpunk/Obsidian dark mode (deep radial gradient backgrounds) paired with Neon Clinical Accents (Cyan, Electric Purple, Crimson) to guide the user's eye to critical data.
- **Glassmorphism:** The patient summary cards and header utilize heavy backdrop blurring and translucent backgrounds, creating a modern, floating glass effect.
- **Micro-animations:** Interactive elements, such as the "Urgent" risk badge, feature sophisticated neon pulse animations to immediately draw attention without being visually abrasive. The entire design prioritizes legibility of complex medical data through structured tables, pill-tags, and clear visual hierarchy (using the `Outfit` font).

## 📄 Example Input and Output

### Input: Urgent Lab Report
A lab report for James Okafor (51M) presenting with confusion and palpitations. The raw document contains critically abnormal values including Potassium 6.8 mEq/L and eGFR 19 mL/min.

### Output: Structured Clinical Reasoning
**1. Extraction (Abbreviated JSON):**
```json
"vital_signs": [
  {"name": {"value": "Potassium", "confidence": 1.0, "evidence_snippet": "Potassium"},
   "value": {"value": "6.8", "confidence": 1.0, "evidence_snippet": "6.8 mEq/L"},
   "status": {"value": "abnormal", "confidence": 1.0, "evidence_snippet": "CRITICAL HIGH ↑↑"}},
  {"name": {"value": "eGFR", "confidence": 1.0, "evidence_snippet": "eGFR"},
   "value": {"value": "19", "confidence": 1.0, "evidence_snippet": "19 mL/min"},
   "status": {"value": "abnormal", "confidence": 1.0, "evidence_snippet": "CRITICAL LOW ↓↓"}}
]
```

**2. Risk Assessment & Action (JSON):**
```json
{
  "risk_level": "Urgent",
  "justification": "Patient has critically elevated potassium (6.8 mEq/L) and severely reduced kidney function (eGFR 19) requiring immediate medical intervention.",
  "driving_factors": [
    "Potassium 6.8 mEq/L (critical high)",
    "eGFR 19 mL/min (Stage 4 CKD)"
  ],
  "recommended_next_step": "Escalate to on-call physician immediately — initiate emergent potassium-lowering protocol and obtain nephrology consultation."
}
```
*In the UI, this is presented as a floating glass card with a pulsing red "🚨 URGENT" badge, where abnormal values are highlighted in crimson, and the recommended next step is placed in a highlighted action box.*
