import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define the model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Step 2: Define a resuable prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {role}. Be concise and practical."),
    ("user", "{question}")
])

# Step 3: Output parser
parser = StrOutputParser()

# Step 4: Chain them together with | operator
chain = prompt_template | llm | parser

# Step 5: Run it
result = chain.invoke({
    "role": "startup advisor", 
    "question": "What are the top 3 mistakes first-time founders make?"
})

print(result)

