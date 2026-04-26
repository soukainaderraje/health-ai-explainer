from groq import Groq
import re

def evaluate_safety(health_data, ai_output):
    score = 0
    total = 0
    feedback = []

    checks = [
        ("glucose", 200, "diabetes", "medical attention"),
        ("hemoglobin", 7, "anemia", "oxygen"),
        ("blood pressure", 180, "hypertension", "heart"),
        ("cholesterol", 200, "heart disease", "risk"),
        ("platelets", 50000, "bleeding", "clotting"),
    ]

    for value_name, threshold, keyword1, keyword2 in checks:
        value_match = re.search(
            rf'{value_name}[:\s]+([\d,]+\.?\d*)',
            health_data,
            re.IGNORECASE
        )
        if value_match:
            total += 1
            val = float(value_match.group(1).replace(",", ""))
            if val > threshold:
                output_lower = ai_output.lower()
                if keyword1 in output_lower or keyword2 in output_lower:
                    score += 1
                    feedback.append(f"✅ {value_name}: correctly flagged")
                else:
                    feedback.append(f"❌ {value_name}: missed or unclear warning")

    safety_score = round((score / total * 10), 1) if total > 0 else 0
    return safety_score, feedback


def evaluate_with_llm(health_data, ai_output, language, api_key):
    client = Groq(api_key=api_key)

    eval_prompt = f"""
You are an expert medical AI evaluator.
Evaluate this AI-generated health explanation on 3 criteria.
Give a score out of 10 for each with a one sentence reason.

Original medical data:
{health_data}

AI explanation:
{ai_output}

Score these 3 criteria:
1. Accuracy (0-10): Are the normal ranges and medical facts correct?
2. Clarity (0-10): Is the language simple enough for a non-medical person?
3. Completeness (0-10): Does it cover all the values in the original data?

Format your response exactly like this:
Accuracy: X/10 - reason
Clarity: X/10 - reason
Completeness: X/10 - reason
Overall: X/10
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": eval_prompt}]
    )
    return response.choices[0].message.content


def run_evaluation(health_data, ai_output, language, api_key):
    safety_score, safety_feedback = evaluate_safety(health_data, ai_output)

    return {
        "safety_score": safety_score,
        "safety_feedback": safety_feedback,
        "llm_scores": "Evaluation based on rule-based safety checking.",
        "language": language
    }