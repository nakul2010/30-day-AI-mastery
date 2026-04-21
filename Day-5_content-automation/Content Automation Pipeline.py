from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("groq_api_key"))

def write_with_tone(topic, content_type, tone, audience, word_count=100):
    tone_instructions = {
        "professional": "formal, authoritative, use data and statistics, third person",
        "conversational": "friendly and warm, use 'you', short sentences, relatable examples",
        "persuasive": "benefit-focused, address pain points, strong call to action",
        "educational": "clear, use analogies, build from simple to complex, use examples"
    }

    prompt = f"""Write a {content_type} about: {topic}

Tone: {tone_instructions.get(tone, tone)}
Target audience: {audience}
Length: approximately {word_count} words

Write directly — no meta-commentary, no "Here is your article" — just the content itself."""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

topic = "Why every small business should start using AI tools?"

for tone in ["professional", "conversational", "persuasive", "educational"]:
    print(f"\n{'='*55}")
    print(f"TONE: {tone.upper()}")
    print(f"\n{'='*55}")
    print(write_with_tone(
        topic=topic,
        content_type="LinkedIn post",
        tone=tone,
        audience="small business owners in India",
        word_count=100
    ))