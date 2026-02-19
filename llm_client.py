# backend/llm_client.py

import os
from groq import Groq

# 自动读取环境变量 GROQ_API_KEY
client = Groq()

def rewrite_note_with_llm(original_note: str, analysis_context: str) -> str:
    prompt = f"""
You are a hospital utilization review documentation specialist.

Rewrite the following clinical note into a Revised HPI that strictly supports inpatient medical necessity.

You MUST follow this exact structure and paragraph order:

Paragraph 1:
- Patient demographics (age, language if available, relevant comorbidities)
- Presenting complaint
- Symptom progression

Paragraph 2:
- Objective findings from the emergency department
- Vital signs (especially respiratory rate and oxygen saturation)
- Oxygen desaturation and supplemental oxygen requirement
- Imaging findings
- Laboratory findings

Paragraph 3:
- Emergency department management
- Blood cultures if obtained
- Intravenous antibiotic initiation
- Admission decision

Paragraph 4 (Summary Paragraph):
- Explicitly summarize admission triggers:
  - Hypoxemia
  - Failure of outpatient therapy
  - Leukocytosis
  - Imaging-confirmed pneumonia
  - Advanced age
- End with strong language such as:
  "warranting inpatient-level management."

Rules:
- Use professional reviewer documentation language.
- Use phrases like "Per ER documentation" when citing evidence.
- Do NOT use bullet points.
- Separate paragraphs with blank lines.
- Do NOT combine everything into a single paragraph.
- Do NOT invent data.
- Do NOT include commentary outside the Revised HPI.

Original Clinical Note:
{original_note}

Supporting Rule Summary:
{analysis_context}

Return ONLY the Revised HPI.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a hospital utilization review documentation specialist writing structured inpatient admission justifications."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1000
    )

    return response.choices[0].message.content.strip()
