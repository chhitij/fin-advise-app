# Install packages
# !pip install -q gradio langchain langchain_groq langgraph yfinance
import gradio as gr
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict, Optional, Dict, List
import re
import yfinance as yf
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === LOAD API KEYS FROM ENVIRONMENT ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

if not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY not found in environment variables!")
if not LANGSMITH_API_KEY:
    logger.warning("⚠️ LANGSMITH_API_KEY not found in environment variables!")

# Configure LangSmith tracing
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "finadvise-app")
    logger.info("✅ LangSmith tracing enabled")


# === STATE ===
class FinanceState(TypedDict):
    user_input: str
    intent: Optional[str]
    data: Optional[dict]
    user_profile: Optional[Dict[str, str]]
    short_term_memory: Optional[Dict[str, str]]
    long_term_memory: Optional[Dict[str, str]]
    hitl_flag: Optional[bool]
    expenses: Optional[List[Dict[str, any]]]

# === LLM ===
try:
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-5-nano",
        temperature=0.3
    )
    logger.info("Using gpt-5-nano model")
except Exception as e:
    logger.warning(f"Failed to initialize gpt-5-nano: {e}. Falling back to gpt-4o-mini")
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.3
    )

# === GLOBAL STORAGE ===
global_user_profile = {}
global_long_term_memory = {}
global_expenses = []
global_short_term_memory = {}  # ✅ NEW: Persist short-term memory across turns

# === NODE FUNCTIONS ===
def detect_intent(state: FinanceState) -> FinanceState:
    user_input = state["user_input"]
    short_term_memory = state.get("short_term_memory", global_short_term_memory.copy())
    long_term_memory = state.get("long_term_memory", {})

    prompt = f"""Classify the user's intent into ONE of: 'profile', 'stock', 'expense', 'budget', 'advice', or 'unknown'.

User input: {user_input}
Previous intent: {short_term_memory.get('previous_intent', 'none')}
Last stock discussed: {short_term_memory.get('last_stock_symbol', 'none')}
Context: {short_term_memory.get('last_context', 'none')}

If user asks a follow-up question like "Is that good?" or "Should I buy it?", and previous intent was 'stock', then intent is still 'stock'.

Intent:"""

    response = llm.invoke(prompt)
    intent = re.search(r"(profile|stock|expense|budget|advice)", response.content.lower())
    intent = intent.group(1) if intent else "unknown"

    # HITL: Check for high-risk keywords
    high_risk_keywords = ["liquidate", "retirement", "all my savings", "entire portfolio", "sell everything"]
    hitl_flag = any(keyword in user_input.lower() for keyword in high_risk_keywords)

    short_term_memory['previous_intent'] = intent
    logger.info(f"Intent detected: {intent}, HITL flag: {hitl_flag}")

    return {**state, "intent": intent, "short_term_memory": short_term_memory, "hitl_flag": hitl_flag}

def collect_user_data(state: FinanceState) -> FinanceState:
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    short_term_memory = state.get("short_term_memory", {})

    prompt = f"""Extract user profile information (age, income, financial goal, risk tolerance) from: {user_input}.
Current profile: {user_profile}.
If no new information is provided, ask a question to gather missing data.
Keep tone empathetic and clear."""

    response = llm.invoke(prompt)
    message = response.content.strip()

    if "age" in message.lower() or "income" in message.lower() or "goal" in message.lower() or "risk" in message.lower():
        for line in message.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                user_profile[key.strip().lower()] = value.strip()

        global_user_profile.update(user_profile)
        logger.info(f"Updated user profile: {user_profile}")
    else:
        short_term_memory['last_question'] = message

    return {**state, "user_profile": user_profile, "data": {"response": message}, "short_term_memory": short_term_memory}

def get_stock_info(state: FinanceState) -> FinanceState:
    """✅ FIXED: Uses context from short-term memory"""
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    short_term_memory = state.get("short_term_memory", global_short_term_memory.copy())

    # Check if this is a follow-up question about a stock
    last_stock = short_term_memory.get('last_stock_symbol', None)

    # Detect if user is asking a follow-up question
    follow_up_patterns = [
        "is that", "is it", "should i", "tell me more", "what about",
        "is this", "good price", "buy it", "worth it", "recommend"
    ]
    is_follow_up = any(pattern in user_input.lower() for pattern in follow_up_patterns)

    if is_follow_up and last_stock:
        # Use the previously discussed stock
        stock_symbol = last_stock
        logger.info(f"Using remembered stock: {stock_symbol}")
        message_prefix = f"📌 *Continuing discussion about {stock_symbol}*\n\n"
    else:
        # Extract new stock symbol
        prompt = f"Extract the stock symbol (e.g., 'AAPL') from: {user_input}. Return ONLY the symbol or 'UNKNOWN'."
        response = llm.invoke(prompt)
        stock_symbol = response.content.strip().upper()
        message_prefix = ""

    if not re.match(r"^[A-Z]{1,5}$", stock_symbol) or stock_symbol == "UNKNOWN":
        logger.warning(f"Invalid stock symbol: {stock_symbol}")
        hint = f" The last stock we discussed was {last_stock}." if last_stock else ""
        return {**state, "data": {"response": f"⚠️ Could not identify stock symbol from '{user_input}'.{hint} Try 'What's the price of AAPL?'"}}

    try:
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        name = info.get('longName', stock_symbol)

        if price:
            # Check if user is asking for advice vs just price
            if is_follow_up or any(word in user_input.lower() for word in ["should", "good", "buy", "invest", "recommend"]):
                # Provide investment advice
                risk_tolerance = user_profile.get('risk tolerance', user_profile.get('risk', 'moderate'))
                age = user_profile.get('age', 'unknown')

                advice_prompt = f"""The user is asking about investing in {name} ({stock_symbol}) at ${price:.2f}.

User profile:
- Risk tolerance: {risk_tolerance}
- Age: {age}

Provide brief, actionable advice (2-3 sentences) on whether this is a good investment for them. Consider:
- Their risk tolerance
- Current valuation
- Diversification
Be clear, empathetic, and avoid jargon."""

                advice_response = llm.invoke(advice_prompt)
                message = f"{message_prefix}📈 **{name} ({stock_symbol})**\nCurrent Price: **${price:.2f}**\n\n💡 **Investment Advice:**\n{advice_response.content.strip()}"
            else:
                # Just provide price
                message = f"{message_prefix}📈 **{name} ({stock_symbol})**\nCurrent Price: **${price:.2f}**"
        else:
            message = f"❌ No price data found for {stock_symbol}"

        logger.info(f"Stock info retrieved: {stock_symbol} = ${price}")

        # ✅ Store in short-term memory for follow-ups
        short_term_memory['last_stock_symbol'] = stock_symbol
        short_term_memory['last_stock_price'] = price
        short_term_memory['last_stock_name'] = name
        short_term_memory['last_context'] = f"Discussed {name} ({stock_symbol}) at ${price:.2f}"
        global_short_term_memory.update(short_term_memory)

    except Exception as e:
        message = f"❌ Error fetching {stock_symbol}: {str(e)}"
        logger.error(f"Stock fetch error: {e}")

    return {**state, "data": {"response": message}, "short_term_memory": short_term_memory}

def track_expenses(state: FinanceState) -> FinanceState:
    user_input = state["user_input"]
    expenses = state.get("expenses", [])

    prompt = f"""Extract the expense amount and category from: "{user_input}"

Return in this EXACT format:
Amount: $XX
Category: category_name

If amount is unclear, return "Amount: UNKNOWN"
If category is unclear, return "Category: Other"

Examples:
"Add $50 for groceries" → Amount: $50, Category: Groceries
"I spent $1200 on rent" → Amount: $1200, Category: Housing"""

    response = llm.invoke(prompt)
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

        message = f"✅ **Expense Added**\n💰 Amount: ${amount:.2f}\n📁 Category: {category}"
        logger.info(f"Expense tracked: ${amount} - {category}")
    else:
        message = "⚠️ I couldn't identify the expense amount. Please specify like '$50 for groceries'"

    return {**state, "data": {"response": message}, "expenses": expenses}

def budget_summary(state: FinanceState) -> FinanceState:
    user_profile = state.get("user_profile", {})
    expenses = state.get("expenses", global_expenses)

    if not expenses:
        income = user_profile.get('income', 'unknown')
        message = f"""📊 **Budget Summary**

⚠️ **No expenses tracked yet!**

You haven't added any expenses. Try:
- "Add $50 for groceries"
- "I spent $1200 on rent"
- "Spent $100 on utilities"

Once you add expenses, I'll show your actual spending breakdown!"""

        if income and income != 'unknown':
            message += f"\n\n💵 Your reported income: {income}"

    else:
        category_totals = {}
        total_spent = 0

        for expense in expenses:
            category = expense['category']
            amount = expense['amount']
            category_totals[category] = category_totals.get(category, 0) + amount
            total_spent += amount

        message = f"📊 **Your Actual Budget Summary**\n\n"
        message += f"**Total Expenses: ${total_spent:.2f}**\n\n"
        message += "**Breakdown by Category:**\n"

        for category, amount in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / total_spent * 100) if total_spent > 0 else 0
            message += f"• {category}: ${amount:.2f} ({percentage:.1f}%)\n"

        income_str = user_profile.get('income', '')
        income_match = re.search(r'\$?([\d,]+)', income_str)

        if income_match:
            income = float(income_match.group(1).replace(',', ''))
            monthly_income = income / 12 if income > 10000 else income
            remaining = monthly_income - total_spent

            message += f"\n**Income Analysis:**\n"
            message += f"• Monthly Income: ${monthly_income:.2f}\n"
            message += f"• Total Spent: ${total_spent:.2f}\n"

            if remaining >= 0:
                message += f"• Remaining: ${remaining:.2f} ✅\n"
            else:
                message += f"• Over Budget: ${abs(remaining):.2f} ⚠️\n"

        message += f"\n**Recent Expenses:**\n"
        for expense in expenses[-5:]:
            message += f"• {expense['date']}: ${expense['amount']:.2f} - {expense['category']}\n"

    logger.info(f"Budget summary generated with {len(expenses)} expenses")

    return {**state, "data": {"response": message}}

def provide_advice(state: FinanceState) -> FinanceState:
    user_input = state["user_input"]
    user_profile = state.get("user_profile", {})
    long_term_memory = state.get("long_term_memory", {})

    prompt = f"""Provide personalized financial advice based on: {user_input}
User profile: {user_profile}
Previous advice: {long_term_memory.get('last_advice', 'none')}
Use clear, empathetic language suitable for users with limited financial literacy."""

    response = llm.invoke(prompt)
    message = f"💡 {response.content.strip()}"

    long_term_memory['last_advice'] = message
    global_long_term_memory['last_advice'] = message

    logger.info("Financial advice provided")

    return {**state, "data": {"response": message}, "long_term_memory": long_term_memory}

def human_in_the_loop(state: FinanceState) -> FinanceState:
    user_input = state["user_input"]

    message = f"""⚠️ **HIGH-RISK QUERY DETECTED**

Your query: "{user_input}"

This query has been flagged as high-risk and requires review by a financial advisor.

🚨 **Important:** Major financial decisions like liquidating retirement accounts should be discussed with a certified financial advisor.

Please wait for expert input before proceeding."""

    logger.warning(f"HITL triggered for: {user_input}")

    return {**state, "data": {"response": message}}

def fallback(state: FinanceState) -> FinanceState:
    return {**state, "data": {"response": "🤔 I didn't understand. Ask about:\n• Stock prices\n• Expense tracking\n• Budget summaries\n• Financial advice"}}

# === ROUTING ===
def get_next_node(state: FinanceState) -> str:
    if state.get("hitl_flag", False):
        return "human_in_the_loop"

    mapping = {
        "profile": "Collect User Data",
        "stock": "Stock Info",
        "expense": "Expense Tracker",
        "budget": "Budget Summary",
        "advice": "Provide Advice"
    }
    return mapping.get(state["intent"], "Fallback")

# === BUILD GRAPH ===
builder = StateGraph(FinanceState)
builder.add_node("Intent Detection", detect_intent)
builder.add_node("Collect User Data", collect_user_data)
builder.add_node("Stock Info", get_stock_info)
builder.add_node("Expense Tracker", track_expenses)
builder.add_node("Budget Summary", budget_summary)
builder.add_node("Provide Advice", provide_advice)
builder.add_node("Human in the Loop", human_in_the_loop)
builder.add_node("Fallback", fallback)

builder.set_entry_point("Intent Detection")
builder.add_conditional_edges(
    "Intent Detection",
    get_next_node,
    {
        "Collect User Data": "Collect User Data",
        "Stock Info": "Stock Info",
        "Expense Tracker": "Expense Tracker",
        "Budget Summary": "Budget Summary",
        "Provide Advice": "Provide Advice",
        "human_in_the_loop": "Human in the Loop",
        "Fallback": "Fallback"
    }
)

finance_bot = builder.compile()

# === GRADIO INTERFACE ===
def chat_with_bot(message, history):
    try:
        state = {
            "user_input": message,
            "intent": None,
            "data": None,
            "user_profile": global_user_profile.copy(),
            "short_term_memory": global_short_term_memory.copy(),  # ✅ Use global memory
            "long_term_memory": global_long_term_memory.copy(),
            "hitl_flag": False,
            "expenses": global_expenses.copy()
        }

        final_state = finance_bot.invoke(state)

        # ✅ Update all global memory
        global_user_profile.update(final_state.get("user_profile", {}))
        global_long_term_memory.update(final_state.get("long_term_memory", {}))
        global_short_term_memory.update(final_state.get("short_term_memory", {}))

        bot_reply = final_state.get("data", {}).get("response", "No response generated.")
        return bot_reply

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return f"❌ Error: {str(e)}"
dark_theme = gr.themes.Monochrome()
# === LAUNCH ===
demo = gr.ChatInterface(
    fn=chat_with_bot,
    title="💰 FinAdvise - Personal Finance Assistant",
    description="Now with working context memory! Try: 'What's Apple stock?' then 'Should I buy it?'",
    examples=[
        "What's Apple stock price?",
        "Is that a good price to buy at?",
        "Add $1200 for rent",
        "Show me a budget summary",
        "Should I liquidate my retirement?"
    ],
    theme=dark_theme
)

print("🚀 FinAdvise is ready!")

if __name__ == "__main__":
    # Render sets the PORT environment variable
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=port,
        share=False  # Don't create Gradio share link on Render
    )
