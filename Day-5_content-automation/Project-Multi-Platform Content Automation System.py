from groq import Groq
from dotenv import load_dotenv
import os 
import json

load_dotenv()
client = Groq(api_key=os.getenv("groq_api_key"))

def llm(prompt, temperature=0.7):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    return response.choices[0].message.content

def content_machine(topic, brand_voice, target_audience):

    print(f"\n Content Machine: '{topic}'\n")

    platforms = {
        "twitter": {
            "format": "tweet under 280 characters, punchy and provocative",
            "count": 3
        },
        "linkedin": {
            "format": "professional post 150-200 words with a hook first line",
            "count": 1
        },
        "instagram_captions": {
            "format": "engaging captain 50-80 words with relevant hashtags",
            "count": 1
        },
        "email_subjects": {
            "format": "email subject line under 50 chars with high open rate",
            "count": 3
        }
    }

    results = {}

    for platform, config in platforms.items():
        content = llm(f"""
        Topic: {topic}
        Brand voice: {brand_voice}
        Target audience: {target_audience}
        Platform: {platform}
        Format: {config['format']}
        Generate: {config['count']} variations
        
        Number each variation. Write only the content, no explanations.
        """, temperature=0.8)
        
        results[platform] = content
        print(f" {platform.upper()} content generated")
    
    # Print full content calendar
    print(f"\n{'='*55}")
    print(" YOUR CONTENT CALENDAR")
    print(f"{'='*55}")

    for platform, content in results.items():
        print(f"\n {platform.upper().replace('_', ' ')}")
        print(f"{'-'*40}")
        print(content)

    return results

content_machine(
    topic="5 ways AI is helping students smarter",
    brand_voice="energetic, practical, youth-friendly",
    target_audience="Indian college students"
)

          
