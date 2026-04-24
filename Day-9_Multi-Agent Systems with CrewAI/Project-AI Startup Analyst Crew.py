from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool
from ddgs import DDGS
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configure Groq via CrewAI's LLM class ─────────────────────────
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("Groq_API_KEY"),
    temperature=0.7
)

@tool("web_search")
def web_search(query: str) -> str:
    """Search web for current market and competitor information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return "\n\n".join([
            f"{r['title']}: {r['body']}" for r in results
        ]) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"

# ── Three Specialized Agents ──────────────────────────────
market_analyst = Agent(
    role="Market Research Analyst",
    goal="Analyze market size, competition, and opportunity for startup ideas",
    backstory="""You are a sharp market analyst who has evaluated 
    500+ startup ideas. You find real market data, identify true competitors,
    and spot market gaps. You are data-driven and specific.""",
    tools=[web_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3
)

business_strategist = Agent(
    role="Startup Business Strategist",
    goal="Design viable business models and go-to-market strategies",
    backstory="""You are a former YC mentor who has helped 50 startups 
    find product-market fit. You design realistic revenue models, 
    identify the ideal customer profile, and create practical 
    90-day launch plans.""",
    tools=[web_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2
)

risk_advisor = Agent(
    role="Startup Risk Advisor",
    goal="Identify critical risks and provide mitigation strategies",
    backstory="""You are a brutally honest startup advisor who has seen 
    1000 startups fail. You identify the real reasons startups die — 
    not obvious risks but hidden ones. You also suggest specific ways 
    to reduce each risk.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2
)

def analyze_startup(idea: str, founder_background: str):
    
    # Task 1: Market Analysis
    market_task = Task(
        description=f"""Analyze the market for this startup idea:
        Idea: {idea}
        Founder background: {founder_background}
        
        Research and provide:
        1. Market size (TAM, SAM, SOM if possible)
        2. Top 3-5 direct competitors with their strengths
        3. Key market trends supporting or threatening this idea
        4. Gaps in current market this idea could fill
        5. Target customer profile with specific demographics""",
        expected_output="Detailed market analysis with specific data points",
        agent=market_analyst
    )
    
    # Task 2: Business Strategy
    strategy_task = Task(
        description=f"""Design a business strategy for:
        Idea: {idea}
        Founder: {founder_background}
        
        Based on the market research, provide:
        1. Recommended revenue model with pricing in INR
        2. Top 3 customer acquisition channels
        3. Minimum viable product (MVP) scope
        4. 90-day launch plan with specific milestones
        5. What makes this defensible long term (moat)""",
        expected_output="Actionable business strategy with 90-day roadmap",
        agent=business_strategist,
        context=[market_task]
    )
    
    # Task 3: Risk Assessment
    risk_task = Task(
        description=f"""Identify critical risks for:
        Idea: {idea}
        
        Based on market analysis and strategy, identify:
        1. Top 3 business risks (market, competition, timing)
        2. Top 3 execution risks (team, tech, operations)  
        3. One existential risk that could kill this startup
        4. Specific mitigation strategy for each risk
        5. Overall verdict: Fund / Improve / Avoid — with reasoning""",
        expected_output="Risk report with verdict and specific mitigation strategies",
        agent=risk_advisor,
        context=[market_task, strategy_task]
    )
    
    # Assemble crew
    crew = Crew(
        agents=[market_analyst, business_strategist, risk_advisor],
        tasks=[market_task, strategy_task, risk_task],
        process=Process.sequential,
        verbose=True
    )
    
    print(f"\n{'='*55}")
    print("🚀 STARTUP ANALYST CREW — ANALYZING YOUR IDEA")
    print(f"{'='*55}\n")
    
    result = crew.kickoff()
    
    # Save report
    with open("startup_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(f"STARTUP ANALYSIS REPORT\n")
        f.write(f"Idea: {idea}\n")
        f.write(f"{'='*55}\n\n")
        f.write(str(result))
    
    print("\n✅ Report saved to startup_analysis_report.txt")
    return result

# Test with your own idea
analyze_startup(
    idea="An AI powered app that helps Indian small shop owners manage inventory and predict stock needs using WhatsApp",
    founder_background="Electronics And Communication Graduate Engineer with knowledge of AI and Python, based in Jaipur"
)