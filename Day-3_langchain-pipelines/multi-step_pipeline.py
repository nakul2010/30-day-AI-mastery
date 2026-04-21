import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define the model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
parser = StrOutputParser()

# Chain 1: Generate a startup idea
idea_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative startup ideation expert."),
    ("user", "Generate ONE specific AI startup idea for the {industry} industry. "
             "One paragraph, include the problem, solution, and target customer.")
])

# Chain 2: Analyze that idea (takes output of chain 1 as input)
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a brutal but fair startup critic like Paul Graham."),
    ("user", "Analyze this startup idea and give it a score out of 10 "
             "with 3 strengths and 3 weaknesses:\n\n{idea}")
])

# Chain 3: Suggest a name (takes output of chain 1 as input)
naming_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a branding expert who names startups."),
    ("user", "Suggest 3 startup names for this idea. "
             "Each name should be short, memorable, and domain-friendly:\n\n{idea}")
])

# Building the chain
idea_chain = idea_prompt | llm | parser
analysis_chain = analysis_prompt | llm | parser
naming_chain = naming_prompt | llm | parser

# Run the full pipeline
print("Generating startup idea...\n")
idea = idea_chain.invoke({"industry": "education"})
print(f"IDEA:\n{idea}\n")

print("Analyzing the idea...\n")
analysis = analysis_chain.invoke({"idea": idea})
print(f"ANALYSIS:\n{analysis}\n")

print("Generating names...\n")
names = naming_chain.invoke({"idea": idea})
print(f" NAMES:\n{names}\n")