from groq import Groq
from rag import get_relevant_info

def run_health_agent(health_data, knowledge_base, language, api_key):
    client = Groq(api_key=api_key)

    # Step 1 - Extract values cleanly
    step1_prompt = f"""
You are a medical data extraction expert.
Extract all medical test values from this text and list them clearly.
Only list the values, nothing else.

Medical data:
{health_data}
"""
    step1 = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": step1_prompt}]
    )
    identified_values = step1.choices[0].message.content

    # Step 2 - Search knowledge base
    relevant_info = get_relevant_info(health_data, knowledge_base)

    # Step 3 - Safety check
    step3_prompt = f"""
You are a medical safety checker.
Given these medical values:
{identified_values}

And these reference ranges:
{relevant_info}

List ONLY the values that are outside normal range.
For each one say: value name, patient result, normal range, and danger level (low/medium/high).
Be direct and concise.
"""
    step3 = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": step3_prompt}]
    )
    safety_check = step3.choices[0].message.content

    # Step 4 - Write final report
    step4_prompt = f"""
You are a helpful and empathetic health educator.
Write a clear patient-friendly report in {language}.

You have:
- Original results: {health_data}
- Identified values: {identified_values}
- Safety concerns: {safety_check}
- Medical reference info: {relevant_info}

Your report must include:
1. Simple explanation of each value
2. Whether each value is normal, high, or low
3. What it means for the patient body
4. General wellness tips
5. Reminder to consult a doctor

Be warm, clear, and reassuring.
"""
    step4 = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": step4_prompt}]
    )
    final_report = step4.choices[0].message.content

    return {
        "identified_values": identified_values,
        "safety_check": safety_check,
        "final_report": final_report
    }