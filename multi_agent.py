from groq import Groq
from rag import get_relevant_info

def run_multi_agent(health_data, knowledge_base, language, api_key):
    client = Groq(api_key=api_key)
    results = {}

    # Agent 1 - Extractor
    agent1_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"You are a medical data extraction specialist. Your only job is to extract and list medical test values clearly and precisely. Respond in {language}."
            },
            {
                "role": "user",
                "content": f"Extract all medical values from this:\n{health_data}"
            }
        ]
    )
    results["extracted"] = agent1_response.choices[0].message.content

    # Agent 2 - Safety Expert
    relevant_info = get_relevant_info(health_data, knowledge_base)

    agent2_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"You are a medical safety expert. Your only job is to identify dangerous values and assess risk levels. Be precise and direct. Respond in {language}."
            },
            {
                "role": "user",
                "content": f"Check these values for dangers:\n{results['extracted']}\n\nReference ranges:\n{relevant_info}"
            }
        ]
    )
    results["safety"] = agent2_response.choices[0].message.content

    # Agent 3 - Doctor
    agent3_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"You are a warm and empathetic doctor explaining results to a patient in {language}. Use simple language. Always recommend consulting a real doctor."
            },
            {
                "role": "user",
                "content": f"""Write a patient report using:
Extracted values: {results['extracted']}
Safety concerns: {results['safety']}
Medical reference: {relevant_info}
Original data: {health_data}"""
            }
        ]
    )
    results["report"] = agent3_response.choices[0].message.content

    # Agent 4 - Reviewer
    agent4_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"You are a senior medical reviewer. Check if the report is accurate, safe, clear, and complete. Give a brief quality score out of 10 and list any missing information. Respond in {language}."
            },
            {
                "role": "user",
                "content": f"Review this patient report:\n{results['report']}"
            }
        ]
    )
    results["review"] = agent4_response.choices[0].message.content

    return results