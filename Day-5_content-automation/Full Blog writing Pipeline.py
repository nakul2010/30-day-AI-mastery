from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────
def llm(prompt, temperature=0.7, system="You are an expert content writer."):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# SAFE JSON PARSER (FIXES YOUR ERROR)
# ─────────────────────────────────────────────
def safe_json_load(text):
    if not text or not text.strip():
        raise ValueError(" Empty response from LLM")

    cleaned = text.strip()

    # remove markdown ```json blocks if present
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]

    # remove 'json' if it exists at start
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("\n JSON PARSE FAILED. RAW RESPONSE:\n")
        print(text)
        raise


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def generate_blog_post(topic, audience, tone="conversational"):
    
    print(f"\n Starting content pipeline for: '{topic}'\n")
    
    # ── STEP 1: Angles ─────────────────────────
    print(" Step 1: Researching angles...")
    angles = llm(
        f"""
        Topic: {topic}
        Audience: {audience}

        Generate 3 unique angles.

        Return ONLY valid JSON:
        ["angle1", "angle2", "angle3"]
        """,
        temperature=0.8,
        system="You ONLY return valid JSON. No text. No explanation."
    )

    angles_list = safe_json_load(angles)
    best_angle = angles_list[0]
    print(f" Best angle selected: {best_angle}")

    # ── STEP 2: Outline ────────────────────────
    print("\n Step 2: Creating outline...")
    outline = llm(
        f"""
        Create a blog outline.

        Topic: {topic}
        Angle: {best_angle}
        Audience: {audience}

        Return ONLY valid JSON:
        {{
            "title": "title",
            "meta_description": "meta",
            "sections": [
                {{"heading": "heading", "key_points": ["p1", "p2"]}}
            ],
            "conclusion_cta": "cta"
        }}
        """,
        temperature=0.3,
        system="You ONLY return valid JSON. No markdown. No explanation."
    )

    outline_data = safe_json_load(outline)

    print(f" Title: {outline_data['title']}")
    print(f" Sections: {len(outline_data['sections'])} sections planned")

    # ── STEP 3: Sections ───────────────────────
    print("\n  Step 3: Writing sections...")

    tone_guide = {
        "conversational": "friendly, use 'you', relatable",
        "professional": "formal, authoritative",
        "educational": "clear, explanatory"
    }

    full_content = ""

    for i, section in enumerate(outline_data['sections'], 1):
        section_content = llm(
            f"""
            Write a blog section.

            Heading: {section['heading']}
            Points: {', '.join(section['key_points'])}
            Tone: {tone_guide.get(tone, tone)}
            Audience: {audience}

            120 words. Start directly.
            """,
            temperature=0.7
        )

        full_content += f"## {section['heading']}\n\n{section_content}\n\n"
        print(f" Section {i}/{len(outline_data['sections'])}")

    # ── STEP 4: Intro + Conclusion ─────────────
    print("\n Step 4: Writing intro and conclusion...")

    intro = llm(
        f"""
        Write an engaging intro for:
        {outline_data['title']}

        Audience: {audience}
        Tone: {tone_guide.get(tone, tone)}

        100 words.
        """,
        temperature=0.8
    )

    conclusion = llm(
        f"""
        Write a conclusion.

        Title: {outline_data['title']}
        CTA: {outline_data['conclusion_cta']}

        Audience: {audience}
        Tone: {tone_guide.get(tone, tone)}

        100 words.
        """,
        temperature=0.7
    )

    final_post = f"# {outline_data['title']}\n\n{intro}\n\n{full_content}"
    final_post += f"## Conclusion\n\n{conclusion}\n\n"
    final_post += f"---\n*Meta: {outline_data['meta_description']}*"

    # ── STEP 5: Quality ────────────────────────
    print("\n Step 5: Quality check...")

    quality = llm(
        f"""
        Rate this blog.

        Return ONLY valid JSON:
        {{
            "clarity_score": 1,
            "engagement_score": 1,
            "seo_score": 1,
            "top_improvement": "tip"
        }}

        Blog:
        {final_post[:500]}
        """,
        temperature=0,
        system="You ONLY return valid JSON."
    )

    quality_data = safe_json_load(quality)

    print(f" Clarity: {quality_data['clarity_score']}/10")
    print(f" Engagement: {quality_data['engagement_score']}/10")
    print(f" SEO: {quality_data['seo_score']}/10")
    print(f" Tip: {quality_data['top_improvement']}")

    return final_post


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
blog = generate_blog_post(
    topic="How AI is changing the job market for Indian college students",
    audience="Indian college students aged 18-22",
    tone="conversational"
)

with open("generated_blog.md", "w", encoding="utf-8") as f:
    f.write(blog)

print("\n FINAL BLOG POST:\n")
print(blog)
print("\n Saved to generated_blog.md")