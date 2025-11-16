# Multi-Agent Financial Advisor System
# Built with LangGraph - Multiple Specialized Agents Collaborating

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, List, Annotated
import operator
import re
import yfinance as yf
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === LOAD API KEYS ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY not found in environment variables!")
if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY not found - Stock Agent will use OpenAI instead")

# Configure LangSmith tracing
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = "finadvise-multiagent"
    logger.info("✅ LangSmith tracing enabled for multi-agent system")


# === MULTI-AGENT STATE ===
class MultiAgentState(TypedDict):
    """State shared across all agents"""
    user_input: str
    user_profile: Optional[Dict[str, str]]
    
    # Agent assignments
    required_agents: List[str]
    completed_agents: Annotated[List[str], operator.add]
    
    # Agent outputs
    stock_analysis: Optional[str]
    financial_advice: Optional[str]
    budget_report: Optional[str]
    risk_assessment: Optional[str]
    
    # Memory
    short_term_memory: Optional[Dict[str, str]]
    long_term_memory: Optional[Dict[str, str]]
    expenses: Optional[List[Dict[str, any]]]
    
    # Control
    next_agent: Optional[str]
    final_response: Optional[str]
    hitl_flag: Optional[bool]


# === GLOBAL STORAGE ===
global_user_profile = {}
global_long_term_memory = {}
global_expenses = []
global_short_term_memory = {}


# === INITIALIZE SPECIALIZED AGENTS ===

# Supervisor Agent - Coordinates all other agents (OpenAI)
supervisor_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.2
)

# Stock Research Agent - Specializes in market analysis (GROQ - Llama 3.3)
# Using Groq's fast inference for real-time stock analysis
try:
    if GROQ_API_KEY:
        stock_agent_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=GROQ_API_KEY,
        )
        logger.info("✅ Stock Agent using Groq Llama-3.3-70b-versatile")
    else:
        raise ValueError("No Groq API key")
except Exception as e:
    logger.warning(f"⚠️ Groq unavailable ({e}), Stock Agent falling back to OpenAI")
    stock_agent_llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.3
    )

# Financial Advisor Agent - Provides investment advice (OpenAI)
advisor_agent_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.3
)

# Budget Analyst Agent - Tracks expenses and budgets (OpenAI)
budget_agent_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.2
)

# Risk Manager Agent - Assesses financial risk (OpenAI)
risk_agent_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.2
)


# === AGENT 1: SUPERVISOR AGENT ===
def supervisor_agent(state: MultiAgentState) -> MultiAgentState:
    """
    Supervisor coordinates which agents should handle the query.
    Acts as the orchestrator of the multi-agent system.
    """
    user_input = state["user_input"]
    short_term_memory = state.get("short_term_memory", {})
    
    logger.info("🎯 SUPERVISOR: Analyzing query and delegating to agents...")
    
    prompt = f"""You are a supervisor coordinating a team of financial expert agents.

User Query: "{user_input}"
Previous Context: {short_term_memory.get('last_context', 'None')}

Available Agents:
1. stock_agent - Stock market research, price analysis, technical indicators
2. advisor_agent - Investment advice, portfolio recommendations, strategies
3. budget_agent - Expense tracking, budget analysis, spending reports
4. risk_agent - Risk assessment, safety checks, compliance verification

Analyze the query and decide which agents are needed.

Rules:
- For stock queries: Use stock_agent + advisor_agent (for price + advice)
- For "should I buy/invest": Use stock_agent + advisor_agent + risk_agent
- For expense tracking: Use budget_agent
- For budget summaries: Use budget_agent
- For financial advice: Use advisor_agent + risk_agent
- For high-risk queries (retirement, liquidate): Use ALL agents

Return ONLY a comma-separated list of agent names needed.
Example: "stock_agent,advisor_agent,risk_agent"
"""

    response = supervisor_llm.invoke(prompt)
    agents_needed = [agent.strip() for agent in response.content.split(",")]
    
    # Check for high-risk keywords
    high_risk_keywords = ["liquidate", "retirement", "all my savings", "entire portfolio", "sell everything"]
    hitl_flag = any(keyword in user_input.lower() for keyword in high_risk_keywords)
    
    if hitl_flag:
        logger.warning("⚠️ SUPERVISOR: High-risk query detected! Flagging for human review.")
        agents_needed = ["stock_agent", "advisor_agent", "budget_agent", "risk_agent"]
    
    logger.info(f"🎯 SUPERVISOR: Delegating to agents: {agents_needed}")
    
    return {
        **state,
        "required_agents": agents_needed,
        "completed_agents": [],
        "hitl_flag": hitl_flag,
        "next_agent": agents_needed[0] if agents_needed else "aggregator"
    }


# === AGENT 2: STOCK RESEARCH AGENT ===
def stock_research_agent(state: MultiAgentState) -> MultiAgentState:
    """
    Specialized agent for stock market research and analysis.
    Provides detailed stock information, trends, and metrics.
    """
    user_input = state["user_input"]
    short_term_memory = state.get("short_term_memory", global_short_term_memory.copy())
    
    logger.info("📊 STOCK AGENT: Analyzing market data...")
    
    # Detect stock symbol
    last_stock = short_term_memory.get('last_stock_symbol', None)
    follow_up_patterns = ["is that", "is it", "should i", "tell me more", "what about", "buy it", "worth it"]
    is_follow_up = any(pattern in user_input.lower() for pattern in follow_up_patterns)
    
    if is_follow_up and last_stock:
        stock_symbol = last_stock
        logger.info(f"📊 STOCK AGENT: Using remembered stock: {stock_symbol}")
    else:
        prompt = f"Extract the stock symbol (e.g., 'AAPL') from: {user_input}. Return ONLY the symbol or 'UNKNOWN'."
        response = stock_agent_llm.invoke(prompt)
        stock_symbol = response.content.strip().upper()
    
    if not re.match(r"^[A-Z]{1,5}$", stock_symbol) or stock_symbol == "UNKNOWN":
        analysis = f"⚠️ Could not identify stock symbol from query. Please specify a valid ticker (e.g., AAPL, TSLA)."
    else:
        try:
            stock = yf.Ticker(stock_symbol)
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            name = info.get('longName', stock_symbol)
            pe_ratio = info.get('trailingPE', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            day_high = info.get('dayHigh', 'N/A')
            day_low = info.get('dayLow', 'N/A')
            
            if price:
                analysis = f"""📊 **Stock Research Report: {name} ({stock_symbol})**

**Market Data:**
- Current Price: ${price:.2f}
- Day Range: ${day_low:.2f} - ${day_high:.2f}
- Market Cap: ${market_cap:,} (if available)
- P/E Ratio: {pe_ratio}

**Technical Analysis:**
The stock is currently trading at ${price:.2f}. """
                
                # Store in memory
                short_term_memory['last_stock_symbol'] = stock_symbol
                short_term_memory['last_stock_price'] = price
                short_term_memory['last_stock_name'] = name
                short_term_memory['last_context'] = f"Analyzed {name} ({stock_symbol}) at ${price:.2f}"
                global_short_term_memory.update(short_term_memory)
                
                logger.info(f"📊 STOCK AGENT: Successfully analyzed {stock_symbol} at ${price:.2f}")
            else:
                analysis = f"❌ No price data available for {stock_symbol}"
                
        except Exception as e:
            analysis = f"❌ Error fetching stock data: {str(e)}"
            logger.error(f"📊 STOCK AGENT: Error - {e}")
    
    return {
        **state,
        "stock_analysis": analysis,
        "short_term_memory": short_term_memory,
        "completed_agents": ["stock_agent"]
    }


# === AGENT 3: FINANCIAL ADVISOR AGENT ===
def financial_advisor_agent(state: MultiAgentState) -> MultiAgentState:
    """
    Specialized agent for investment advice and financial planning.
    Provides personalized recommendations based on user profile.
    """
    user_input = state["user_input"]
    user_profile = state.get("user_profile", global_user_profile.copy())
    stock_analysis = state.get("stock_analysis", "")
    short_term_memory = state.get("short_term_memory", {})
    
    logger.info("💼 ADVISOR AGENT: Formulating investment advice...")
    
    # Extract key info from stock analysis if available
    stock_symbol = short_term_memory.get('last_stock_symbol', 'N/A')
    stock_price = short_term_memory.get('last_stock_price', 'N/A')
    
    prompt = f"""You are a certified financial advisor providing personalized investment advice.

User Query: {user_input}
Stock Analysis: {stock_analysis}

User Profile:
- Age: {user_profile.get('age', 'unknown')}
- Income: {user_profile.get('income', 'unknown')}
- Risk Tolerance: {user_profile.get('risk tolerance', 'moderate')}
- Financial Goal: {user_profile.get('goal', 'long-term growth')}

Provide clear, actionable investment advice (3-4 sentences) considering:
1. User's risk tolerance and age
2. Current market conditions
3. Diversification principles
4. Long-term vs short-term strategy

Be empathetic, clear, and avoid jargon. Start with "💼 Investment Recommendation:"
"""

    response = advisor_agent_llm.invoke(prompt)
    advice = response.content.strip()
    
    logger.info("💼 ADVISOR AGENT: Investment advice generated")
    
    return {
        **state,
        "financial_advice": advice,
        "completed_agents": ["advisor_agent"]
    }


# === AGENT 4: BUDGET ANALYST AGENT ===
def budget_analyst_agent(state: MultiAgentState) -> MultiAgentState:
    """
    Specialized agent for expense tracking and budget analysis.
    Manages user spending and provides financial insights.
    """
    user_input = state["user_input"]
    expenses = state.get("expenses", global_expenses.copy())
    user_profile = state.get("user_profile", {})
    
    logger.info("💰 BUDGET AGENT: Analyzing expenses and budget...")
    
    # Check if this is expense tracking or budget summary
    is_expense_tracking = any(word in user_input.lower() for word in ["add", "spent", "paid", "expense"])
    
    if is_expense_tracking:
        # Extract expense details
        prompt = f"""Extract expense amount and category from: "{user_input}"

Return in EXACT format:
Amount: $XX
Category: category_name

If unclear, return "Amount: UNKNOWN" or "Category: Other"
"""
        response = budget_agent_llm.invoke(prompt)
        response_text = response.content.strip()
        
        amount_match = re.search(r"Amount:\s*\$?(\d+(?:\.\d{2})?)", response_text)
        category_match = re.search(r"Category:\s*(.+)", response_text)
        
        if amount_match:
            amount = float(amount_match.group(1))
            category = category_match.group(1).strip() if category_match else "Other"
            
            expense = {
                "amount": amount,
                "category": category,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "description": user_input
            }
            expenses.append(expense)
            global_expenses.append(expense)
            
            report = f"""💰 **Expense Tracked Successfully**

✅ Amount: ${amount:.2f}
📁 Category: {category}
📅 Date: {expense['date']}

Total expenses tracked: {len(global_expenses)}"""
            
            logger.info(f"💰 BUDGET AGENT: Tracked ${amount} - {category}")
        else:
            report = "⚠️ Could not identify expense amount. Please specify like '$50 for groceries'"
    else:
        # Generate budget summary
        if not expenses:
            report = """💰 **Budget Summary**

⚠️ No expenses tracked yet!

Start tracking by saying:
- "Add $50 for groceries"
- "I spent $1200 on rent"
"""
        else:
            category_totals = {}
            total_spent = 0
            
            for expense in expenses:
                cat = expense['category']
                amt = expense['amount']
                category_totals[cat] = category_totals.get(cat, 0) + amt
                total_spent += amt
            
            report = f"""💰 **Budget Analysis Report**

**Total Expenses: ${total_spent:.2f}**

**Breakdown by Category:**
"""
            for category, amount in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
                percentage = (amount / total_spent * 100) if total_spent > 0 else 0
                report += f"\n• {category}: ${amount:.2f} ({percentage:.1f}%)"
            
            report += f"\n\n**Recent Transactions:**"
            for expense in expenses[-3:]:
                report += f"\n• {expense['date']}: ${expense['amount']:.2f} - {expense['category']}"
            
            logger.info(f"💰 BUDGET AGENT: Generated summary for {len(expenses)} expenses")
    
    return {
        **state,
        "budget_report": report,
        "expenses": expenses,
        "completed_agents": ["budget_agent"]
    }


# === AGENT 5: RISK MANAGER AGENT ===
def risk_manager_agent(state: MultiAgentState) -> MultiAgentState:
    """
    Specialized agent for risk assessment and safety checks.
    Validates recommendations and flags potential issues.
    """
    user_input = state["user_input"]
    stock_analysis = state.get("stock_analysis", "")
    financial_advice = state.get("financial_advice", "")
    user_profile = state.get("user_profile", {})
    hitl_flag = state.get("hitl_flag", False)
    
    logger.info("🛡️ RISK AGENT: Performing risk assessment...")
    
    if hitl_flag:
        assessment = f"""🛡️ **HIGH-RISK DECISION DETECTED**

⚠️ Your query: "{user_input}"

**Risk Level: CRITICAL**

This decision involves significant financial implications:
- Potential for large losses
- Long-term impact on financial security
- Requires expert consultation

🚨 **Recommendation:** Please consult with a certified financial advisor before proceeding.

**Next Steps:**
1. Schedule consultation with financial planner
2. Review your complete financial situation
3. Consider tax implications
4. Evaluate alternative options
"""
        logger.warning("🛡️ RISK AGENT: CRITICAL risk level - HITL recommended")
    else:
        # Analyze risk of current recommendations
        prompt = f"""You are a risk management specialist reviewing financial recommendations.

User Query: {user_input}
User Risk Tolerance: {user_profile.get('risk tolerance', 'moderate')}
Stock Analysis: {stock_analysis[:200]}
Financial Advice: {financial_advice[:200]}

Assess the risk level (LOW/MEDIUM/HIGH) and provide:
1. Risk level assessment
2. Key risk factors (2-3 points)
3. Risk mitigation suggestions (if applicable)

Keep it brief (2-3 sentences). Start with "🛡️ Risk Assessment:"
"""
        
        response = risk_agent_llm.invoke(prompt)
        assessment = response.content.strip()
        
        logger.info("🛡️ RISK AGENT: Risk assessment completed")
    
    return {
        **state,
        "risk_assessment": assessment,
        "completed_agents": ["risk_agent"]
    }


# === AGGREGATOR NODE ===
def aggregator_node(state: MultiAgentState) -> MultiAgentState:
    """
    Aggregates outputs from all agents into a cohesive response.
    """
    logger.info("🔄 AGGREGATOR: Combining agent outputs...")
    
    parts = []
    
    if state.get("stock_analysis"):
        parts.append(state["stock_analysis"])
    
    if state.get("financial_advice"):
        parts.append("\n\n" + state["financial_advice"])
    
    if state.get("risk_assessment"):
        parts.append("\n\n" + state["risk_assessment"])
    
    if state.get("budget_report"):
        parts.append(state["budget_report"])
    
    if not parts:
        final_response = "🤔 I couldn't process your request. Please try asking about stocks, expenses, or financial advice."
    else:
        final_response = "".join(parts)
        
        # Add multi-agent footer
        agents_used = state.get("completed_agents", [])
        final_response += f"\n\n---\n*Analysis by: {', '.join([a.replace('_', ' ').title() for a in agents_used])}*"
    
    logger.info("🔄 AGGREGATOR: Final response generated")
    
    return {
        **state,
        "final_response": final_response
    }


# === ROUTING LOGIC ===
def route_to_next_agent(state: MultiAgentState) -> str:
    """
    Determines which agent to call next, or if we should aggregate.
    """
    required = set(state.get("required_agents", []))
    completed = set(state.get("completed_agents", []))
    remaining = required - completed
    
    if not remaining:
        logger.info("🔄 ROUTING: All agents completed, moving to aggregation")
        return "aggregator"
    
    next_agent = list(remaining)[0]
    logger.info(f"🔄 ROUTING: Next agent -> {next_agent}")
    return next_agent


# === BUILD MULTI-AGENT GRAPH ===
workflow = StateGraph(MultiAgentState)

# Add all agent nodes
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("stock_agent", stock_research_agent)
workflow.add_node("advisor_agent", financial_advisor_agent)
workflow.add_node("budget_agent", budget_analyst_agent)
workflow.add_node("risk_agent", risk_manager_agent)
workflow.add_node("aggregator", aggregator_node)

# Set entry point
workflow.set_entry_point("supervisor")

# Add conditional routing from supervisor to specialized agents
workflow.add_conditional_edges(
    "supervisor",
    route_to_next_agent,
    {
        "stock_agent": "stock_agent",
        "advisor_agent": "advisor_agent",
        "budget_agent": "budget_agent",
        "risk_agent": "risk_agent",
        "aggregator": "aggregator"
    }
)

# Each agent routes back through the router
workflow.add_conditional_edges(
    "stock_agent",
    route_to_next_agent,
    {
        "stock_agent": "stock_agent",
        "advisor_agent": "advisor_agent",
        "budget_agent": "budget_agent",
        "risk_agent": "risk_agent",
        "aggregator": "aggregator"
    }
)

workflow.add_conditional_edges(
    "advisor_agent",
    route_to_next_agent,
    {
        "stock_agent": "stock_agent",
        "advisor_agent": "advisor_agent",
        "budget_agent": "budget_agent",
        "risk_agent": "risk_agent",
        "aggregator": "aggregator"
    }
)

workflow.add_conditional_edges(
    "budget_agent",
    route_to_next_agent,
    {
        "stock_agent": "stock_agent",
        "advisor_agent": "advisor_agent",
        "budget_agent": "budget_agent",
        "risk_agent": "risk_agent",
        "aggregator": "aggregator"
    }
)

workflow.add_conditional_edges(
    "risk_agent",
    route_to_next_agent,
    {
        "stock_agent": "stock_agent",
        "advisor_agent": "advisor_agent",
        "budget_agent": "budget_agent",
        "risk_agent": "risk_agent",
        "aggregator": "aggregator"
    }
)

# Aggregator goes to END
workflow.add_edge("aggregator", END)

# Compile the multi-agent system
multi_agent_system = workflow.compile()

logger.info("✅ Multi-Agent System compiled successfully!")


# === GRADIO INTERFACE ===
def chat_with_multiagent(message, history):
    """
    Chat interface for the multi-agent system.
    """
    try:
        logger.info(f"\n{'='*60}\n🎤 USER: {message}\n{'='*60}")
        
        state = {
            "user_input": message,
            "user_profile": global_user_profile.copy(),
            "required_agents": [],
            "completed_agents": [],
            "stock_analysis": None,
            "financial_advice": None,
            "budget_report": None,
            "risk_assessment": None,
            "short_term_memory": global_short_term_memory.copy(),
            "long_term_memory": global_long_term_memory.copy(),
            "expenses": global_expenses.copy(),
            "next_agent": None,
            "final_response": None,
            "hitl_flag": False
        }
        
        # Run the multi-agent system
        final_state = multi_agent_system.invoke(state)
        
        # Update global state
        global_user_profile.update(final_state.get("user_profile", {}))
        global_short_term_memory.update(final_state.get("short_term_memory", {}))
        global_long_term_memory.update(final_state.get("long_term_memory", {}))
        
        bot_reply = final_state.get("final_response", "No response generated.")
        
        logger.info(f"{'='*60}\n🤖 SYSTEM: Response delivered\n{'='*60}\n")
        
        return bot_reply
        
    except Exception as e:
        logger.error(f"❌ Error in multi-agent system: {e}")
        return f"❌ Error: {str(e)}"


# === LAUNCH GRADIO APP ===
dark_theme = gr.themes.Monochrome()

demo = gr.ChatInterface(
    fn=chat_with_multiagent,
    title="💰 FinAdvise - Multi-Agent Multi-Model Financial Advisory System",
    description="🤖 **Hybrid Multi-Agent Architecture**: Supervisor (GPT-4o-mini) coordinates Stock Researcher (Groq Llama-3.3-70b), Financial Advisor (GPT-4o-mini), Budget Analyst (GPT-4o-mini), and Risk Manager (GPT-4o-mini) working together!",
    examples=[
        "What's Apple stock price?",
        "Should I invest in Tesla?",
        "Add $50 for groceries",
        "Show me a budget summary",
        "Should I liquidate my retirement account?"
    ],
    theme=dark_theme
)

print("🚀 Multi-Agent Multi-Model FinAdvise System is ready!")
print("📊 Agents:")
print("   - Supervisor: OpenAI GPT-4o-mini")
print("   - Stock Researcher: Groq Llama-3.3-70b (fast inference)")
print("   - Financial Advisor: OpenAI GPT-4o-mini")
print("   - Budget Analyst: OpenAI GPT-4o-mini")
print("   - Risk Manager: OpenAI GPT-4o-mini")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    
    print(f"\n{'='*60}")
    print(f"Environment PORT: {os.getenv('PORT', 'NOT SET (using default 7860)')}")
    print(f"Binding to: 0.0.0.0:{port}")
    print(f"{'='*60}\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )
