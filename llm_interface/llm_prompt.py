import os
from openai import OpenAI



system_prompt = """
You are a medical assistant trained to diagnose diseases using a Bayesian network. The network contains the following nodes:

- Causes: Flu, COVID-19
- Symptoms: Fever, Cough, Fatigue
- Treatments: Rest, Medication
- Outcome: Recovery

Your task is to:
1. Ask questions to identify the presence of symptoms (Fever, Cough, Fatigue).
2. Based on the conversation, infer and generate evidence in JSON format like: {"Fever": 1, "Cough": 0, "Fatigue": None}. Use `None` if information is not provided.
3. Ask 2-3 follow-up questions to gather additional relevant information about the symptoms or treatments if applicable. For example:
   - If the user mentions symptoms like "I have a fever," confirm its severity or associated symptoms like fatigue or cough.
   - If the user mentions treatments like "I tried medication," ask for their response to it or any ongoing issues.
4. Avoid irrelevant questions unrelated to diagnosis.
5. Always prioritize information needed for diagnosis over treatments unless treatments are contextually relevant.
6. End the conversation once sufficient evidence is collected for diagnosis.
"""



client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)



history = []
def chat_with_gpt3(history, newprompt):
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": newprompt})
    history = messages
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    response = response.choices[0].message.content

    history.append({"role": "assistant", "content": response})
    
    return history, response



while True:
    user_input = input("User:")
    if user_input == "stop":
        break
    history, result = chat_with_gpt3(history, user_input)
    print(result)
