from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("groq_api_key"))

messages = [
    {"role": "system", "content": "You are a sharp AI mentor."}
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7
    )

    reply = response.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})

    print(f"AI: {reply}\n")


                
        
