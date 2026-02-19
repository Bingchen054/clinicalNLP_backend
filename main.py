from llm_client import rewrite_note_with_llm
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
from typing import List, Optional, Dict, Any
import pdfplumber

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    note: str


# -----------------------------
# Text Normalization
# -----------------------------
def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
    text = re.sub(r"\b([A-Za-z]+)\.\s", r"\1 ", text)
    text = text.replace("SpO 2", "SpO2")
    text = text.replace("O 2", "O2")
    return text.strip()


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_age(text: str) -> Optional[int]:
    m = re.search(r"(\d{2,3})[- ]?year[- ]?old", text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    m = re.search(r"Age[:\s]+(\d{1,3})", text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None


def extract_o2_sat(text: str) -> Optional[int]:
    # Normalize spaced letters like "S P O 2"
    text = re.sub(r"S\s*P\s*O\s*2", "SpO2", text, flags=re.IGNORECASE)

    # Standard patterns
    patterns = [
        r"(?:SpO2|O2 sat|oxygen saturation)[^0-9]{0,5}([0-9]{2,3})",
        r"([0-9]{2,3})\s*%"  # fallback pattern
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                val = int(m.group(1))
                if 50 <= val <= 100:
                    return val
            except:
                continue

    return None



def extract_creatinine(text: str) -> Optional[float]:
    # Fix "CREATININE." pattern
    text = re.sub(r"CREATININE\.", "CREATININE", text, flags=re.IGNORECASE)

    m = re.search(r"\bcreatinine\b[^0-9]{0,15}([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            if 0.1 <= val <= 20:
                return val
        except:
            return None

    return None



def extract_troponin(text: str) -> Optional[float]:
    m = re.search(r"\btroponin\b[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            if 0 <= val <= 100:
                return val
        except:
            return None
    return None


def extract_sbp(text: str) -> Optional[int]:
    # Standard BP format: 120/80
    m = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", text)
    if m:
        try:
            val = int(m.group(1))
            if 60 <= val <= 250:
                return val
        except:
            pass

    # Handle broken table format like:
    # 15 2/ 68  → 152/68
    m = re.search(r"(\d{2})\s*(\d)\s*/\s*(\d{2})", text)
    if m:
        try:
            val = int(m.group(1) + m.group(2))
            if 60 <= val <= 250:
                return val
        except:
            pass

    # Fallback: look for SBP labeled explicitly
    m = re.search(r"\bSBP[:\s]*?(\d{2,3})\b", text, flags=re.IGNORECASE)
    if m:
        try:
            val = int(m.group(1))
            if 60 <= val <= 250:
                return val
        except:
            pass

    return None



def detect_pneumonia(text: str) -> bool:
    text = text.lower()

    patterns = [
        r"\bpneumonia\b",
        r"\binfiltrate(s)?\b",
        r"\bconsolidation\b",
        r"\bopacity\b",
        r"\blobar pneumonia\b",
        r"\bconsistent with pneumonia\b",
        r"\brll pneumonia\b",
        r"\bright lower lobe\b.*\bpneumonia\b"
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def detect_iv_antibiotics(text: str) -> bool:
    # Normalize case
    text = text.lower()

    # Broad IV antibiotic detection
    patterns = [
        r"\biv\b.*\bantibiotic",
        r"\bintravenous\b.*\bantibiotic",
        r"\bceftriaxone\b",
        r"\bvancomycin\b",
        r"\bzosyn\b",
        r"\bcefepime\b",
        r"\bpiperacillin\b",
        r"\bmeropenem\b"
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


def detect_hemodynamic_instability_phrase(text: str) -> bool:
    text = text.lower()

    patterns = [
        r"\bhypotension\b",
        r"\bhypotensive\b",
        r"\bhemodynamic instability\b",
        r"\bshock\b",
        r"\bunstable\b",
        r"\bsbp\s*<\s*90\b",
        r"\bsystolic\s*<\s*90\b",
        r"\bmap\s*<\s*65\b",
        r"\bpressors?\b",
        r"\bnorepinephrine\b",
        r"\bvasopressor\b"
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


# -----------------------------
# Guideline Parser
# -----------------------------
def parse_guideline_thresholds(guideline_text: str) -> Dict[str, Any]:
    rules = {
        "o2_sat_threshold": 90,
        "troponin_threshold": 0.04,
        "creatinine_threshold": 1.5,
        "sbp_threshold": 90
    }

    m = re.search(r"(?:o2|saturation)[^\d]{0,20}(\d{2})", guideline_text, flags=re.IGNORECASE)
    if m:
        try:
            rules["o2_sat_threshold"] = int(m.group(1))
        except:
            pass

    return rules


# -----------------------------
# Rule Engine
# -----------------------------
def evaluate_rules(features: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    score = 0
    justifications: List[str] = []
    checklist: List[Dict[str, Any]] = []

    # ---------------- Oxygen Saturation ----------------
    if features.get("o2_sat") is not None:
        if features["o2_sat"] < rules["o2_sat_threshold"]:
            status = "Met"
            score += 3
            justifications.append(f"Hypoxia documented (SpO2 {features['o2_sat']}%).")
            confidence = 0.95
        else:
            status = "Partial"
            confidence = 0.75
    else:
        status = "Missing"
        confidence = 0.9

    checklist.append({
        "criteria": "Oxygen saturation",
        "status": status,
        "evidence": f"SpO2: {features.get('o2_sat')}" if features.get("o2_sat") else "No oxygen saturation documented.",
        "guideline": f"MCG suggests admission if SpO2 < {rules['o2_sat_threshold']}%.",
        "action": "Document oxygen saturation and need for supplemental oxygen." if status != "Met" else "No additional action needed.",
        "confidence": confidence
    })

    # ---------------- Pneumonia ----------------
    if features.get("pneumonia"):
        status = "Met"
        score += 2
        justifications.append("Radiographic evidence of pneumonia documented.")
        confidence = 0.9
    else:
        status = "Missing"
        confidence = 0.85

    checklist.append({
        "criteria": "Radiographic evidence of pneumonia",
        "status": status,
        "evidence": "Imaging suggests pneumonia." if features.get("pneumonia") else "No imaging documentation found.",
        "guideline": "MCG requires objective imaging confirmation.",
        "action": "Document imaging findings clearly." if status != "Met" else "No additional action needed.",
        "confidence": confidence
    })

    # ---------------- Creatinine ----------------
    if features.get("creatinine") is not None:
        status = "Met"
        confidence = 0.8
    else:
        status = "Missing"
        confidence = 0.8

    checklist.append({
        "criteria": "Creatinine",
        "status": status,
        "evidence": f"Creatinine: {features.get('creatinine')}" if features.get("creatinine") else "No creatinine value documented.",
        "guideline": "Renal dysfunction increases severity and may support admission.",
        "action": "Document renal function." if status != "Met" else "No additional action needed.",
        "confidence": confidence
    })

    # ---------------- Troponin ----------------
    if features.get("troponin") is not None:
        status = "Met"
        confidence = 0.8
    else:
        status = "Missing"
        confidence = 0.75

    checklist.append({
        "criteria": "Troponin documentation",
        "status": status,
        "evidence": f"Troponin: {features.get('troponin')}" if features.get("troponin") else "No troponin result documented.",
        "guideline": "Cardiac involvement should be ruled out in elderly patients.",
        "action": "Document troponin if clinically indicated." if status != "Met" else "No additional action needed.",
        "confidence": confidence
    })

    # ---------------- Blood Pressure ----------------
    if features.get("sbp") is not None:
        status = "Met"
        confidence = 0.85
    else:
        status = "Missing"
        confidence = 0.85

    checklist.append({
        "criteria": "Blood pressure",
        "status": status,
        "evidence": f"SBP: {features.get('sbp')}" if features.get("sbp") else "No systolic blood pressure documented.",
        "guideline": "Hemodynamic instability is an admission criterion.",
        "action": "Document blood pressure and hemodynamic status." if status != "Met" else "No additional action needed.",
        "confidence": confidence
    })

    # ---------------- IV Antibiotics ----------------
    if features.get("iv_antibiotics"):
        status = "Met"
        justifications.append("IV antibiotics documented.")
        confidence = 0.9
    else:
        status = "Missing"
        confidence = 0.9

    checklist.append({
        "criteria": "IV antibiotics",
        "status": status,
        "evidence": "IV antibiotics documented." if features.get("iv_antibiotics") else "No intravenous antibiotic therapy documented.",
        "guideline": "MCG supports inpatient care when IV antibiotics are required.",
        "action": "Document IV antibiotic administration." if status != "Met" else "No additional action needed.",
        "confidence": confidence
    })

    # Admission Level
    if score >= 6:
        level = "Inpatient - strongly supported"
    elif score >= 3:
        level = "Inpatient - possible"
    else:
        level = "Observation / outpatient"

    return {
        "score": score,
        "level": level,
        "justifications": justifications,
        "missingCriteria": checklist
    }




# -----------------------------
# Text-only Analyze
# -----------------------------
@app.post("/analyze")
def analyze(data: AnalyzeRequest):
    note = normalize_text(data.note or "")

    features = {
        "age": extract_age(note),
        "o2_sat": extract_o2_sat(note),
        "creatinine": extract_creatinine(note),
        "troponin": extract_troponin(note),
        "sbp": extract_sbp(note),
        "pneumonia": detect_pneumonia(note),
        "iv_antibiotics": detect_iv_antibiotics(note),
        "hemodynamic_phrase": detect_hemodynamic_instability_phrase(note)
    }

    default_rules = {
        "o2_sat_threshold": 90,
        "troponin_threshold": 0.04,
        "creatinine_threshold": 1.5,
        "sbp_threshold": 90
    }

    results = evaluate_rules(features, default_rules)

    try:
        summary = "\n".join(results["justifications"])
        revised_note = rewrite_note_with_llm(note, summary)
    except Exception as e:
        revised_note = note + "\n\nLLM error: " + str(e)

    return {
        "revisedNote": revised_note,
        "missingCriteria": results["missingCriteria"],
        "score": results["score"],
        "level": results["level"]
    }


# -----------------------------
# Analyze with Guideline
# -----------------------------
@app.post("/analyze-with-guideline")
async def analyze_with_guideline(
    doctor_note: str = Form(...),
    guideline: UploadFile = File(...)
):
    doctor_note = normalize_text(doctor_note)

    guideline_text = ""
    with pdfplumber.open(guideline.file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                guideline_text += page_text + "\n"

    guideline_text = normalize_text(guideline_text)

    rules = parse_guideline_thresholds(guideline_text)

    features = {
        "age": extract_age(doctor_note),
        "o2_sat": extract_o2_sat(doctor_note),
        "creatinine": extract_creatinine(doctor_note),
        "troponin": extract_troponin(doctor_note),
        "sbp": extract_sbp(doctor_note),
        "pneumonia": detect_pneumonia(doctor_note),
        "iv_antibiotics": detect_iv_antibiotics(doctor_note),
        "hemodynamic_phrase": detect_hemodynamic_instability_phrase(doctor_note)
    }

    results = evaluate_rules(features, rules)

    try:
        analysis_context = f"""
Guideline Summary:
{guideline_text[:4000]}

Extracted Rule Justifications:
{"; ".join(results["justifications"])}
"""
        revised_note = rewrite_note_with_llm(doctor_note, analysis_context)
    except Exception as e:
        revised_note = doctor_note + "\n\nLLM error: " + str(e)

    return {
        "revisedNote": revised_note,
        "missingCriteria": results["missingCriteria"],
        "score": results["score"],
        "level": results["level"]
    }
