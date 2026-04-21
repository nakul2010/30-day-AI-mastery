from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm_creative = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8)
llm_precise = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
parser = JsonOutputParser()

def analyze_business_idea(idea: str, industry: str, target_market: str):
    
    # Pipeline Step 1: Market Analysis
    market_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a market research analyst. Return ONLY valid JSON."),
        ("user", """Analyze the market for this idea:
        Idea: {idea}
        Industry: {industry}
        Target: {target_market}
        
        Return JSON:
        {{
            "market_size": "large/medium/small",
            "competition_level": "high/medium/low",
            "top_competitors": ["name1", "name2", "name3"],
            "market_trend": "growing/stable/declining",
            "entry_difficulty": "easy/medium/hard"
        }}""")
    ])
    
    # Pipeline Step 2: Revenue Model
    revenue_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business model expert. Return ONLY valid JSON."),
        ("user", """Suggest revenue models for:
        Idea: {idea}
        Market Analysis: {market_analysis}
        
        Return JSON:
        {{
            "recommended_model": "SaaS/marketplace/freemium/one-time",
            "pricing_suggestion": "specific price range in INR",
            "monthly_revenue_potential": "conservative estimate",
            "time_to_first_revenue": "weeks/months"
        }}""")
    ])
    
    # Pipeline Step 3: Action Plan
    action_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a startup execution coach."),
        ("user", """Create a 30-day action plan for:
        Idea: {idea}
        Revenue Model: {revenue_model}
        
        Give 4 specific weekly milestones.
        Be brutally practical — no fluff.""")
    ])

    print(f"\n{'='*55}")
    print(f"AI BUSINESS ANALYST REPORT")
    print(f"{'='*55}")
    
    # Run Step 1
    print("\n📊 Analyzing market...")
    market_chain = market_prompt | llm_precise | parser
    market = market_chain.invoke({
        "idea": idea,
        "industry": industry,
        "target_market": target_market
    })
    print(f"Market Size: {market['market_size'].upper()}")
    print(f"Competition: {market['competition_level'].upper()}")
    print(f"Trend: {market['market_trend'].upper()}")
    print(f"Top Competitors: {', '.join(market['top_competitors'])}")

    # Run Step 2
    print("\n💰 Building revenue model...")
    revenue_chain = revenue_prompt | llm_precise | parser
    revenue = revenue_chain.invoke({
        "idea": idea,
        "market_analysis": str(market)
    })
    print(f"Model: {revenue['recommended_model']}")
    print(f"Pricing: {revenue['pricing_suggestion']}")
    print(f"Revenue Potential: {revenue['monthly_revenue_potential']}")

    # Run Step 3
    print("\n🗓 Creating action plan...")
    action_chain = action_prompt | llm_creative | JsonOutputParser()
    action_chain_str = action_prompt | llm_creative
    from langchain_core.output_parsers import StrOutputParser
    action_chain_final = action_prompt | llm_creative | StrOutputParser()
    action_plan = action_chain_final.invoke({
        "idea": idea,
        "revenue_model": str(revenue)
    })
    print(f"\n{action_plan}")
    print(f"\n{'='*55}")

# Test it with your own idea
analyze_business_idea(
    idea="An AI tool that helps small Indian businesses write GST invoices and file returns automatically",
    industry="Fintech",
    target_market="Small business owners and freelancers in India"
)
     