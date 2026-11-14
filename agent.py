import os
import csv
import time
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# Load environment variables from .env
load_dotenv()

# 1. Define Custom Tools (AI-5 Requirements)
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

# 2. Configure Groq LLM and Custom System Persona (AI-6 Customisation)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

system_persona = (
    "You are an Enterprise E-Commerce & Financial Compliance Assistant. "
    "Always verify product availability and base price using the search tool first. "
    "If a product is 'Out of Stock' or 'Product not found in catalog.', state that clearly and DO NOT calculate a total cost. "
    "If in stock, always use the calculate_total_cost tool to apply discounts and 8% tax accurately."
)

# Create agent using updated langchain.agents import
agent = create_agent(llm, tools=tools, system_prompt=system_persona)

# 3. Build the 10-Case Evaluation Harness (AI-7 Requirements)
test_cases = [
    {"input": "Find the price of a laptop and calculate total with 10% discount", "expected_substring": "$1166.40"},
    {"input": "Search for headphones and apply a 20% discount", "expected_substring": "$129.60"},
    {"input": "What is the price of a smartwatch?", "expected_substring": "not found"},
    {"input": "Look up a keyboard and calculate cost with 0% discount", "expected_substring": "$86.40"},
    {"input": "Find a monitor price and apply 15% discount", "expected_substring": "Out of Stock"},
    {"input": "Check laptop price and calculate total with 50% discount", "expected_substring": "$648.00"},
    {"input": "Search for headphones and apply 50% discount", "expected_substring": "$81.00"},
    {"input": "Find price for tablet", "expected_substring": "not found"},
    {"input": "Search keyboard and calculate total with 25% discount", "expected_substring": "$64.80"},
    {"input": "Look up monitor and check cost with 10% discount", "expected_substring": "Out of Stock"}
]

def run_evaluation():
    print("\n🚀 Starting Evaluation Harness (Groq Llama-3.3-70B) with Intermediate Step Tracing...")
    passed = 0
    results = []

    for i, test in enumerate(test_cases):
        print(f"\n--- Running Test Case {i+1} ---")
        print(f"Prompt: {test['input']}")
        try:
            response = agent.invoke({"messages": [HumanMessage(content=test["input"])]})
            messages = response["messages"]

            # Trace intermediate steps (Tool calls & Tool observations)
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [Tool Call] -> {tc['name']} | Args: {tc['args']}")
                elif msg.type == "tool":
                    print(f"  [Tool Output] -> {msg.content}")

            # Extract final answer (Groq returns standard string content)
            output = str(messages[-1].content).strip()
            print(f"  [Final Answer] -> {output}")

            is_pass = test["expected_substring"].lower() in output.lower()
            if is_pass:
                passed += 1
                print("Result: ✅ PASS")
            else:
                print(f"Result: ❌ FAIL (Expected '{test['expected_substring']}')")

            results.append({
                "test_case": i + 1,
                "input": test["input"],
                "output": output,
                "status": "PASS" if is_pass else "FAIL"
            })
        except Exception as e:
            print(f"Result: ⚠️ ERROR ({e})")
            results.append({
                "test_case": i + 1,
                "input": test["input"],
                "output": str(e),
                "status": "ERROR"
            })

        # Brief 2s pause between test cases
        if i < len(test_cases) - 1:
            time.sleep(2)

    pass_rate = (passed / len(test_cases)) * 100
    print(f"\n================================")
    print(f"📊 Evaluation Complete: {passed}/{len(test_cases)} Passed ({pass_rate}%)")
    print(f"================================\n")

    # Export results to CSV
    with open("custom_evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_case", "input", "output", "status"])
        writer.writeheader()
        writer.writerows(results)
    print("📁 Results saved to custom_evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()