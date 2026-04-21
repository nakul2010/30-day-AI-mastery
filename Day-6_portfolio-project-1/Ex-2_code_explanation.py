from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv("groq_api_key")
if not api_key:
    raise ValueError("Missing `groq_api_key` in environment. Add it to your .env file.")

client = Groq(api_key=api_key)


def llm(prompt: str, temperature: float = 0.7) -> str:
    """Call Groq chat completion and return text content."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


def generate_platform_content(topic: str, brand_voice: str, target_audience: str,
                              platform: str, format_desc: str, count: int):
    """Generate parsed JSON list for one platform, with fallback on parse failure."""
    prompt = f"""
You are a social media content assistant.

Topic: {topic}
Brand voice: {brand_voice}
Target audience: {target_audience}
Platform: {platform}
Format: {format_desc}
Generate: {count} variations

IMPORTANT:
- Return STRICT JSON only.
- Use this exact schema:
{{
  "variations": ["...", "..."]
}}
- Do not include markdown, numbering, or extra commentary.
"""

    try:
        raw = llm(prompt, temperature=0.8)
        parsed = json.loads(raw)

        if "variations" not in parsed or not isinstance(parsed["variations"], list):
            raise ValueError("Invalid JSON schema returned by model.")

        # Ensure exactly `count` items at most
        return parsed["variations"][:count], None

    except json.JSONDecodeError:
        return [], f"{platform}: Model did not return valid JSON."
    except Exception as e:
        return [], f"{platform}: {str(e)}"


def save_content_calendar(results: dict, output_file: str = "content_calendar.json") -> None:
    """Save generated calendar to JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved content calendar to: {output_file}")


def content_machine(topic: str, brand_voice: str, target_audience: str):
    print(f"\nContent Machine: '{topic}'\n")

    platforms = {
        "twitter": {
            "format": "Tweet under 280 characters, punchy and provocative",
            "count": 3
        },
        "linkedin": {
            "format": "Professional post (150-200 words) with a strong first-line hook",
            "count": 1
        },
        "instagram_captions": {
            "format": "Engaging caption (50-80 words) with relevant hashtags",
            "count": 1
        },
        "email_subjects": {
            "format": "Email subject line under 50 characters with high open-rate potential",
            "count": 3
        }
    }

    results = {}
    errors = []

    for platform, config in platforms.items():
        variations, err = generate_platform_content(
            topic=topic,
            brand_voice=brand_voice,
            target_audience=target_audience,
            platform=platform,
            format_desc=config["format"],
            count=config["count"]
        )

        results[platform] = variations
        if err:
            errors.append(err)
            print(f"{platform.upper()}: generation failed")
        else:
            print(f"{platform.upper()}: content generated")

    print(f"\n{'=' * 55}")
    print("YOUR CONTENT CALENDAR")
    print(f"{'=' * 55}")

    for platform, items in results.items():
        print(f"\n{platform.upper().replace('_', ' ')}")
        print("-" * 40)
        if not items:
            print("No content generated.")
            continue
        for i, text in enumerate(items, start=1):
            print(f"{i}. {text}")

    if errors:
        print(f"\n{'!' * 55}")
        print("WARNINGS")
        print(f"{'!' * 55}")
        for e in errors:
            print(f"- {e}")

    save_content_calendar(results)
    return results


if __name__ == "__main__":
    content_machine(
        topic="5 ways AI is helping students study smarter",
        brand_voice="energetic, practical, youth-friendly",
        target_audience="Indian college students"
    )
