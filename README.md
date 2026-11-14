# Autonomous E-Commerce & Financial Compliance Agent

An autonomous multi-step AI agent built with **LangChain** and **Groq (`openai/gpt-oss-120b`)**, featuring a specialized enterprise compliance persona, custom tool orchestration, intermediate step tracing, and a 10-case automated evaluation harness.

---

## 🎯 Project Overview

This project fulfills the **AI-5, AI-6, and AI-7** engineering milestones by transitioning a standard LLM into a deterministic, tool-using autonomous workflow:

1. **Multi-Step Autonomous Agent (AI-5):** Uses LangChain's `create_agent` ReAct loop to dynamically chain multiple tools (`search_product_database` $\rightarrow$ `calculate_total_cost`) to resolve multi-step user queries.
2. **Custom System Persona & Guardrails (AI-6):** Configured with an **Enterprise E-Commerce & Financial Compliance Assistant** system persona that enforces strict business rules—preventing cost calculations on out-of-stock or missing items and mandating verified tool usage for all tax/discount math.
3. **Evaluation Harness & Step Tracing (AI-7):** Benchmarked against a 10-case test suite covering happy paths, boundary discounts, missing catalog items, and out-of-stock guardrails, exporting structured pass/fail metrics to `custom_evaluation_results.csv`.

---

## 🛠️ Architecture & Tools

| Component | Description |
| :--- | :--- |
| **Orchestration** | LangChain (`langchain.agents.create_agent`) |
| **LLM Backbone** | Groq API (`openai/gpt-oss-120b`, `temperature=0`) |
| **Tool 1: `search_product_database`** | Queries the internal product catalog to retrieve base pricing and real-time inventory status (`In Stock`, `Low Stock`, `Out of Stock`). |
| **Tool 2: `calculate_total_cost`** | Computes final customer pricing by applying dynamic percentage discounts followed by a mandatory 8% sales tax. |
| **Observability** | Logs every intermediate `[Tool Call]`, `[Tool Output]`, and `[Final Answer]` in the terminal during execution. |

---

## 🚀 Local Setup & Execution (Windows / VS Code)

### 1. Clone the Repository
```powershell
git clone [https://github.com/pasiketansaig5genai-blip/Agentic-AI.git](https://github.com/pasiketansaig5genai-blip/Agentic-AI.git)
cd Agentic-AI