from groq import Groq
from rag import get_relevant_info

def run_multi_agent(health_data, knowledge_base, language, api_key):
    client = Groq(api_key=api_key)
    results = {}

    # Get relevant medical info from RAG
    relevant_info = get_relevant_info(health_data, knowledge_base)

    # Call 1 — Extract values + Safety check combined
    call1_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"You are a medical data specialist. Respond in {language}."
            },
            {
                "role": "user",
                "content": f"""Do two things with these medical results:

1. EXTRACTED VALUES: List all medical test values clearly
2. SAFETY ASSESSMENT: Check each value against these reference ranges and identify dangerous values with risk level (low/medium/high)

Reference ranges:
{relevant_info}

Medical results:
{health_data}

Format your response with clear sections:
EXTRACTED VALUES:
[list here]

SAFETY ASSESSMENT:
[list here]"""
            }
        ]
    )
    call1_text = call1_response.choices[0].message.content

    # Split the response
    if "SAFETY ASSESSMENT:" in call1_text:
        parts = call1_text.split("SAFETY ASSESSMENT:")
        results["extracted"] = parts[0].replace("EXTRACTED VALUES:", "").strip()
        results["safety"] = parts[1].strip()
    else:
        results["extracted"] = call1_text
        results["safety"] = "No dangerous values detected."

    # Call 2 — Doctor report + Quality review combined
    call2_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"You are a warm and empathetic doctor explaining results to a patient in {language}. Always recommend consulting a real doctor."
            },
            {
                "role": "user",
                "content": f"""Write a patient-friendly health report then review it.

Extracted values: {results['extracted']}
Safety concerns: {results['safety']}
Medical reference: {relevant_info}
Original data: {health_data}

Format your response with clear sections:
PATIENT REPORT:
[warm friendly explanation of each value, normal/high/low, wellness tips, doctor reminder]

QUALITY REVIEW:
[brief quality score out of 10 and any missing information]"""
            }
        ]
    )
    call2_text = call2_response.choices[0].message.content

    # Split the response
    if "QUALITY REVIEW:" in call2_text:
        parts = call2_text.split("QUALITY REVIEW:")
        results["report"] = parts[0].replace("PATIENT REPORT:", "").strip()
        results["review"] = parts[1].strip()
    else:
        results["report"] = call2_text
        results["review"] = "Quality review not available."

    return results