💰 FinAdvise - AI-Powered Personal Finance Assistant

An intelligent financial advisor chatbot built with **LangGraph**, **LangChain**, and **OpenAI GPT** that helps users manage their finances, track expenses, analyze stocks, and receive personalized financial advice.

## 🎯 Purpose

FinAdvise is designed to democratize financial literacy by providing:
- **Real-time stock price information** using Yahoo Finance
- **Expense tracking and budgeting** with automatic categorization
- **Personalized financial advice** based on user profiles
- **Context-aware conversations** that remember previous discussions
- **Human-in-the-loop safety** for high-risk financial decisions
- **LangSmith tracing** for monitoring and debugging AI interactions

## ✨ Key Features

### 1. **Stock Analysis**
- Get real-time stock prices for any US company
- Contextual follow-up questions (e.g., "Should I buy it?")
- Personalized investment advice based on risk tolerance and age

### 2. **Expense Tracking**
- Add expenses with natural language (e.g., "Add $50 for groceries")
- Automatic categorization (Housing, Food, Transportation, etc.)
- Track spending patterns over time

### 3. **Budget Management**
- View spending breakdown by category
- Compare income vs. expenses
- Get alerts when over budget
- See recent transaction history

### 4. **Financial Advice**
- Personalized recommendations based on your profile
- Risk-appropriate investment strategies
- Budget optimization tips
- Savings goal planning

### 5. **Smart Memory System**
- **Short-term memory**: Remembers context within conversation
- **Long-term memory**: Stores user profile and financial goals
- **Expense history**: Tracks all transactions

### 6. **Safety Features**
- **Human-in-the-Loop (HITL)**: Flags high-risk queries like "liquidate retirement"
- **Risk assessment**: Provides advice based on user's risk tolerance
- **Transparent tracking**: All AI interactions logged in LangSmith

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- LangSmith API key (optional, for tracking)

### Installation

1. **Clone the repository:**
```bash
cd d:\genrative-ai-repo\stock-analysis
```

2. **Create conda environment:**
```bash
conda create -n finapp python=3.10
conda activate finapp
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create `.env` file:**
```env
OPENAI_API_KEY=sk-your-openai-key-here
LANGSMITH_API_KEY=lsv2_pt_your-langsmith-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=finadvise-app
```

5. **Run the app:**
```bash
python app.py
```

6. **Open in browser:**
- Local: http://127.0.0.1:7860
- Public: Check terminal for Gradio share link

## 📊 Sample Questions

### Stock Analysis
```
✅ "What's Apple stock price?"
✅ "How much is TSLA trading at?"
✅ "Tell me about Microsoft stock"
✅ "Should I buy AAPL?" (after asking about Apple)
✅ "Is that a good price?" (follow-up question)
```

### Expense Tracking
```
✅ "Add $50 for groceries"
✅ "I spent $1200 on rent"
✅ "Spent $100 on utilities yesterday"
✅ "Add $45.99 for dinner"
✅ "I paid $80 for gas"
```

### Budget Summary
```
✅ "Show me a budget summary"
✅ "What's my spending breakdown?"
✅ "How much have I spent this month?"
✅ "Am I over budget?"
```

### Financial Advice
```
✅ "Should I invest in stocks or bonds?"
✅ "How much should I save each month?"
✅ "What's a good emergency fund amount?"
✅ "Is it a good time to invest?"
✅ "Should I pay off debt or invest?"
```

### User Profile
```
✅ "I'm 30 years old"
✅ "My income is $60,000 per year"
✅ "My goal is to save for retirement"
✅ "I have moderate risk tolerance"
✅ "I want to buy a house in 5 years"
```

### Follow-up Questions (Context Awareness)
```
✅ "What's Google stock?" → "Should I buy it?"
✅ "Add $200 for groceries" → "Show me my food expenses"
✅ "Tell me about Tesla" → "Is it worth investing in?"
```

### High-Risk Queries (HITL Triggers)
```
⚠️ "Should I liquidate my retirement account?"
⚠️ "I want to invest all my savings in crypto"
⚠️ "Should I sell everything?"
⚠️ "I want to use my entire portfolio for one stock"
```

## 🏗️ Architecture

### LangGraph Workflow
```
User Input
    ↓
[Intent Detection] ──→ Classifies: profile/stock/expense/budget/advice
    ↓
[Routing Logic] ──→ Selects appropriate node
    ↓
[Action Node] ──→ Executes task (fetch stock, track expense, etc.)
    ↓
[Response] ──→ Returns to user
```

### State Management
```python
FinanceState = {
    "user_input": str,              # Current message
    "intent": str,                  # Detected intent
    "data": dict,                   # Response data
    "user_profile": dict,           # Age, income, risk tolerance
    "short_term_memory": dict,      # Last stock, previous intent
    "long_term_memory": dict,       # Historical advice
    "hitl_flag": bool,              # Safety trigger
    "expenses": list                # Tracked expenses
}
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **AI Framework** | LangChain, LangGraph |
| **LLM** | OpenAI GPT-4o-mini (fallback: gpt-5-nano) |
| **Stock Data** | yfinance |
| **UI** | Gradio (ChatInterface) |
| **Monitoring** | LangSmith |
| **Environment** | python-dotenv |

## 📈 LangSmith Tracking

View all AI interactions, prompts, and responses:
1. Go to https://smith.langchain.com
2. Select project: **finadvise-app**
3. View traces for:
   - All LLM calls
   - State transitions
   - Token usage
   - Latency metrics

## 🎨 Customization

### Change Theme
```python
# In app.py, line 437
dark_theme = gr.themes.Soft()  # Options: Monochrome, Soft, Glass, Base
```

### Adjust LLM Temperature
```python
# In app.py, line 56
llm = ChatOpenAI(
    temperature=0.3  # Lower = more consistent, Higher = more creative
)
```

### Add New Intent
1. Update `detect_intent()` prompt
2. Add new node function
3. Update routing in `get_next_node()`
4. Add conditional edge in graph builder

## 🔒 Security

- ✅ API keys stored in `.env` (never committed)
- ✅ High-risk queries flagged for human review
- ✅ User data stored in memory only (not persisted)
- ✅ All AI calls logged for audit trail

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
- Check `.env` file exists in project root
- Verify key format: `OPENAI_API_KEY=sk-...`

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Stock data errors
- Yahoo Finance may rate-limit requests
- Use valid stock symbols (AAPL, TSLA, GOOGL, etc.)

### LangSmith not tracking
- Verify `LANGCHAIN_TRACING_V2=true` in `.env`
- Check API key is valid
- Ensure internet connection

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more financial indicators (P/E ratio, dividends)
- Persistent storage (database integration)
- Multi-language support
- Voice input/output
- Portfolio optimization algorithms

## 📧 Contact

For questions or support, open an issue on GitHub.

---

**Built with ❤️ using LangChain and LangGraph**