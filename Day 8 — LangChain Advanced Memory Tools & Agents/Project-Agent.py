from groq import Groq
import os
from ddgs import DDGS
import json
import datetime
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("groq_api_key"))

class ConversationMemory:
    """
    Manual memory implementation.
    This is exactly what LangChain's ConversationBufferMemory does internally.
    """
    def __init__(self, system_prompt="You are a helpful AI assistant."):
        self.system_prompt = system_prompt
        self.messages = []
    
    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})
    
    def add_ai_message(self, message):
        self.messages.append({"role": "assistant", "content": message})
    
    def get_messages(self):
        return [{"role": "system", "content": self.system_prompt}] + self.messages
    
    def clear(self):
        self.messages = []
    
    def get_summary(self):
        return f"Conversation has {len(self.messages)} messages stored."

def chat_with_memory(client, memory, user_input):
    memory.add_user_message(user_input)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=memory.get_messages(),
        temperature=0.7
    )
    
    reply = response.choices[0].message.content
    memory.add_ai_message(reply)
    return reply

# ── Test Memory ───────────────────────────────────────────
print("="*55)
print("CONVERSATION WITH MEMORY")
print("="*55)

memory = ConversationMemory(
    system_prompt="You are a helpful AI mentor teaching about AI."
)

# Turn 1
reply1 = chat_with_memory(client, memory, 
    "Hi, my name is Nakul and I'm doing a 30-day AI mastery program.")
print(f"Turn 1: {reply1}\n")

# Turn 2
reply2 = chat_with_memory(client, memory, 
    "What is RAG and why should I learn it?")
print(f"Turn 2: {reply2}\n")

# Turn 3 — tests if it remembers Turn 1
reply3 = chat_with_memory(client, memory, 
    "What's my name and what am I doing?")
print(f"Turn 3: {reply3}\n")

print(f"📝 {memory.get_summary()}")



# ── Define Tools as Simple Python Functions ───────────────
# Step 2 - Tools From Scratch
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        output = ""
        for r in results:
            output += f"Title: {r['title']}\nSummary: {r['body']}\n\n"
        return output.strip()
    except Exception as e:
        return f"Search failed: {e}"

def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    try:
        allowed = set('0123456789+-*/.() ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"Result: {result}"
        return "Error: Invalid expression"
    except Exception as e:
        return f"Error: {e}"

def get_current_date(query: str = "") -> str:
    """Get today's date."""
    return datetime.datetime.now().strftime("Today is %A, %B %d, %Y")

def word_counter(text: str) -> str:
    """Count words in text."""
    words = len(text.split())
    chars = len(text)
    return f"Words: {words}, Characters: {chars}"

# Tool registry — agent looks up tools here
TOOLS = {
    "web_search": {
        "function": web_search,
        "description": "Search the web for current info. Input: search query string."
    },
    "calculator": {
        "function": calculator,
        "description": "Evaluate math. Input: math expression like '500 * 1.12 ** 3'."
    },
    "get_current_date": {
        "function": get_current_date,
        "description": "Get today's date. Input: any string."
    },
    "word_counter": {
        "function": word_counter,
        "description": "Count words and characters. Input: the text to count."
    }
}

def run_tool(tool_name: str, tool_input: str) -> str:
    """Execute a tool by name."""
    if tool_name not in TOOLS:
        return f"Tool '{tool_name}' not found. Available: {list(TOOLS.keys())}"
    return TOOLS[tool_name]["function"](tool_input)


# Step 3 - ReAct Agent Loop From Scratch
def build_system_prompt():
    tool_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    ])
    
    return f"""You are a helpful AI agent with access to tools.

AVAILABLE TOOLS:
{tool_descriptions}

INSTRUCTIONS:
To use a tool, respond in this EXACT format:
THOUGHT: [your reasoning about what to do]
ACTION: [tool_name]
INPUT: [input for the tool]

When you have enough information to answer, respond in this EXACT format:
THOUGHT: [your final reasoning]
FINAL ANSWER: [your complete answer to the user]

Always start with THOUGHT. Always end with FINAL ANSWER.
Never make up information — use tools to find real answers."""

def run_react_agent(question: str, max_iterations: int = 5):
    """
    The ReAct loop:
    Thought → Action → Observation → Thought → ... → Final Answer
    """
    
    print(f"\n{'='*55}")
    print(f"Question: {question}")
    print(f"{'='*55}")
    
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": question}
    ]
    
    for iteration in range(max_iterations):
        print(f"\n🔄 Iteration {iteration + 1}")
        
        # Get agent's next move
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.0,
            max_tokens=500
        )
        
        agent_output = response.choices[0].message.content
        print(f"\n🤖 Agent:\n{agent_output}")
        
        # Check if agent has final answer
        if "FINAL ANSWER:" in agent_output:
            final = agent_output.split("FINAL ANSWER:")[-1].strip()
            print(f"\n✅ DONE: {final}")
            return final
        
        # Parse the action
        if "ACTION:" in agent_output and "INPUT:" in agent_output:
            try:
                action_line = [l for l in agent_output.split('\n') 
                              if l.startswith("ACTION:")][0]
                input_line = [l for l in agent_output.split('\n') 
                             if l.startswith("INPUT:")][0]
                
                tool_name = action_line.replace("ACTION:", "").strip()
                tool_input = input_line.replace("INPUT:", "").strip()
                
                # Run the tool
                print(f"\n🔧 Running tool: {tool_name}")
                print(f"   Input: {tool_input}")
                
                observation = run_tool(tool_name, tool_input)
                print(f"   Result: {observation[:200]}...")
                
                # Add to conversation so agent sees the result
                messages.append({
                    "role": "assistant", 
                    "content": agent_output
                })
                messages.append({
                    "role": "user", 
                    "content": f"OBSERVATION: {observation}\n\nContinue reasoning."
                })
                
            except Exception as e:
                print(f"⚠️ Parse error: {e}")
                break
        else:
            # Agent responded without using tools
            print(f"\n✅ Answer: {agent_output}")
            return agent_output
    
    return "Max iterations reached."

# ── Test the Agent ────────────────────────────────────────
# Test 1: Needs calculator
run_react_agent(
    "If I invest ₹50,000 at 12% annual return for 3 years compounded annually, how much will I have?"
)

# Test 2: Needs web search
run_react_agent(
    "What are the most popular AI agent frameworks right now?"
)

# Test 3: Multi-tool — needs date + search
run_react_agent(
    "What is today's date and what major AI news happened recently?"
)

# Step 4 - Full Research Agent with Memory 
def research_agent_with_memory():
    """
    Complete agent combining:
    - ReAct reasoning loop
    - Conversation memory
    - Web search + calculator tools
    """
    
    memory = ConversationMemory(
        system_prompt=build_system_prompt()
    )
    
    print("\n" + "="*55)
    print("🔬 PERSONAL AI RESEARCH AGENT")
    print("Commands: 'quit' to exit, 'memory' to see history")
    print("="*55 + "\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Session ended!")
            break
        if user_input.lower() == "memory":
            print(f"\n📝 {memory.get_summary()}")
            for msg in memory.messages[-4:]:
                role = "You" if msg["role"] == "user" else "Agent"
                print(f"{role}: {msg['content'][:100]}...")
            continue
        
        # Add to memory
        memory.add_user_message(user_input)
        
        # Simple single-step response with tool capability
        messages = memory.get_messages()
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.0,
            max_tokens=800
        )
        
        agent_reply = response.choices[0].message.content
        
        # Check if it wants to use a tool
        if "ACTION:" in agent_reply and "INPUT:" in agent_reply:
            try:
                action_line = [l for l in agent_reply.split('\n') 
                              if l.startswith("ACTION:")][0]
                input_line = [l for l in agent_reply.split('\n') 
                             if l.startswith("INPUT:")][0]
                
                tool_name = action_line.replace("ACTION:", "").strip()
                tool_input = input_line.replace("INPUT:", "").strip()
                
                print(f"🔍 Searching: {tool_input}")
                observation = run_tool(tool_name, tool_input)
                
                # Get final answer with observation
                messages.append({"role": "assistant", "content": agent_reply})
                messages.append({
                    "role": "user",
                    "content": f"OBSERVATION: {observation}\n\nNow give your FINAL ANSWER."
                })
                
                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=600
                )
                
                final_reply = final_response.choices[0].message.content
                if "FINAL ANSWER:" in final_reply:
                    final_reply = final_reply.split("FINAL ANSWER:")[-1].strip()
                    
            except Exception:
                final_reply = agent_reply
        else:
            final_reply = agent_reply
            if "FINAL ANSWER:" in final_reply:
                final_reply = final_reply.split("FINAL ANSWER:")[-1].strip()
        
        memory.add_ai_message(final_reply)
        print(f"\nAgent: {final_reply}\n")

research_agent_with_memory()