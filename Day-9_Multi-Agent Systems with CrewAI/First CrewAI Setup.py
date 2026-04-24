from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configure Groq via CrewAI's LLM class ─────────────────────────
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("Groq_API_KEY"),
    temperature=0.7
)

# ── Define Tools ──────────────────────────────────────────
@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for current information on any topic."""

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No results found."
        return "\n\n".join([
            f"Title: {r['title']}\nInfo: {r['body']}"
            for r in results
        ])
    except Exception as e:
        return f"Search failed: {e}"

@tool("word_counter")
def word_counter(text: str) -> str:
    """Count the number of words in a text."""
    return f"Word count: {len(text.split())}"


# ── Define Agent ──────────────────────────────────────────
# ── Agent 1: Researcher ───────────────────────────────────
researcher = Agent(
    role="Senior Market Researcher",
    goal="Find accurate, current information and synthesize key insights",
    backstory="""You are an experienced market researcher with 10 years 
    of experience analyzing tech markets, especially in India. 
    You find facts, not opinions. You cite specific numbers when possible.""",
    tools=[web_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3
)

# ── Agent 2: Writer ───────────────────────────────────────
writer = Agent(
    role="Expert Content Writer",
    goal="Transform research into compelling, clear, well-structured content",
    backstory="""You are a professional writer who specializes in making 
    complex topics simple. Your writing is engaging, structured, and 
    always tailored to the target audience. You never pad content with fluff.""",
    tools=[word_counter],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2
)

# ── Agent 3: Editor ───────────────────────────────────────
editor = Agent(
    role="Senior Content Editor",
    goal="Review and improve content for clarity, accuracy and impact",
    backstory="""You are a meticulous editor with high standards. 
    You improve structure, fix weak arguments, ensure consistency, 
    and make sure every sentence earns its place.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2
)


# Define Tasks and Run the Crew ─────────────────────────────────────────
def run_content_crew(topic: str, audience: str):
    
    # ── Task 1: Research ──────────────────────────────────
    research_task = Task(
        description=f"""Research this topic thoroughly: {topic}
        
        Find:
        1. Key facts and current statistics
        2. Main trends and developments
        3. Real examples and case studies
        4. Relevance to: {audience}
        
        Search the web for current information.
        Produce a detailed research summary with specific facts.""",
        expected_output="A structured research summary with facts, stats, and examples",
        agent=researcher
    )
    
    # ── Task 2: Write ─────────────────────────────────────
    write_task = Task(
        description=f"""Using the research provided, write a compelling article about: {topic}
        
        Requirements:
        - Target audience: {audience}
        - Length: 400-500 words
        - Include: strong opening hook, 3 main sections, actionable conclusion
        - Tone: conversational but informative
        - Use specific facts from the research
        
        Do not make up statistics — only use what was researched.""",
        expected_output="A complete 400-500 word article ready for publishing",
        agent=writer,
        context=[research_task]  # writer sees researcher's output
    )
    
    # ── Task 3: Edit ──────────────────────────────────────
    edit_task = Task(
        description=f"""Edit and improve the article provided.
        
        Check for:
        1. Clarity — is every sentence clear?
        2. Structure — does it flow logically?
        3. Engagement — will {audience} find this compelling?
        4. Accuracy — are claims supported?
        
        Provide the final polished version of the article.
        Also give a 2-line editor's note on what you improved.""",
        expected_output="Final polished article + editor's note on improvements made",
        agent=editor,
        context=[write_task]  # editor sees writer's output
    )
    
    # ── Assemble and Run Crew ─────────────────────────────
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, write_task, edit_task],
        process=Process.sequential,  # research → write → edit
        verbose=True
    )
    
    print(f"\n{'='*55}")
    print(f"🚀 CREW STARTING: {topic}")
    print(f"{'='*55}\n")
    
    result = crew.kickoff()
    return result

# Run it
output = run_content_crew(
    topic="How AI agents are changing jobs in India in 2025",
    audience="Indian college students and fresh graduates"
)

print(f"\n{'='*55}")
print("📄 FINAL OUTPUT:")
print(f"{'='*55}")
print(output)