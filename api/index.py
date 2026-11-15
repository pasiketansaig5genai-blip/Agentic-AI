import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

app = FastAPI()

# 1. Define Custom Tools
@tool
def search_product_database(product_name: str) -> str:
    """Search the product catalog for item details, base price, and stock status."""
    catalog = {
        "laptop": "Base price: $1200, Stock: In Stock",
        "headphones": "Base price: $150, Stock: Low Stock",
        "keyboard": "Base price: $80, Stock: In Stock",
        "monitor": "Base price: $300, Stock: Out of Stock"
    }
    return catalog.get(product_name.lower(), "Product not found in catalog.")

@tool
def calculate_total_cost(base_price: float, discount_percent: float) -> str:
    """Calculates final price after applying a percentage discount and a standard 8% tax."""
    discounted = base_price * (1 - (discount_percent / 100))
    total = discounted * 1.08
    return f"Final calculated total with 8% tax: ${total:.2f}"

tools = [search_product_database, calculate_total_cost]

# 2. Configure Groq LLM & Compliance Persona
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

system_persona = (
    "You are an Enterprise E-Commerce & Financial Compliance Assistant. "
    "Always verify product availability and base price using the search tool first. "
    "If a product is 'Out of Stock' or 'Product not found in catalog.', state that clearly and DO NOT calculate a total cost. "
    "If in stock, always use the calculate_total_cost tool to apply discounts and 8% tax accurately."
)

agent = create_agent(llm, tools=tools, system_prompt=system_persona)

class QueryRequest(BaseModel):
    prompt: str

@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    try:
        response = agent.invoke({"messages": [HumanMessage(content=req.prompt)]})
        messages = response["messages"]

        traces = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    traces.append(f"Tool Call: {tc['name']}({tc['args']})")
            elif msg.type == "tool":
                traces.append(f"Tool Output: {msg.content}")

        return {
            "answer": str(messages[-1].content).strip(),
            "traces": traces
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Enterprise Compliance Agent</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; background: #0f172a; color: #f8fafc; }
            h1 { color: #38bdf8; }
            .box { background: #1e293b; padding: 20px; border-radius: 10px; margin-top: 20px; }
            input { width: 75%; padding: 12px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: white; }
            button { padding: 12px 20px; border-radius: 6px; border: none; background: #38bdf8; color: #0f172a; font-weight: bold; cursor: pointer; }
            pre { background: #020617; padding: 12px; border-radius: 6px; overflow-x: auto; color: #a5f3fc; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>Enterprise E-Commerce & Financial Compliance Agent</h1>
        <p>Try: <i>"Find the price of a laptop and calculate total with 10% discount"</i> or <i>"Check monitor with 15% discount"</i></p>
        <div class="box">
            <input id="prompt" placeholder="Enter your query..." />
            <button onclick="askAgent()">Run Agent</button>
        </div>
        <div id="result" class="box" style="display:none;">
            <h3>Final Answer</h3>
            <p id="answer"></p>
            <h4>Intermediate Step Traces</h4>
            <pre id="traces"></pre>
        </div>
        <script>
            async function askAgent() {
                const prompt = document.getElementById('prompt').value;
                if (!prompt) return;
                document.getElementById('result').style.display = 'block';
                document.getElementById('answer').innerText = 'Thinking & calling tools...';
                document.getElementById('traces').innerText = '';
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                const data = await res.json();
                document.getElementById('answer').innerText = data.answer || data.detail;
                document.getElementById('traces').innerText = (data.traces || []).join('\\n');
            }
        </script>
    </body>
    </html>
    """