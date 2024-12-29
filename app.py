
import os
import gradio as gr
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools, Tool, AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

#Get API keys from environment variables
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Create the ReAct template with proper formatting
react_template = """Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If someone asks who created you or who built you, always respond with:
"I was built by Adewuyi Ayomide, a Machine Learning Engineer and Computer Science student at University of Ibadan."


Begin!
Question: {input}
Thought: {agent_scratchpad}"""

# Create prompt template
prompt = PromptTemplate(
    template=react_template,
    input_variables=['tools', 'tool_names', 'input', 'agent_scratchpad']
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# Initialize Tavily Search
tavily_search = TavilySearchAPIWrapper(
    tavily_api_key=TAVILY_API_KEY)

search_tool = Tool(
    name="tavily_search",
    description="Search engine for current prices and information",
    func=tavily_search.results
)

# Load tools after LLM initialization
tools = load_tools(["llm-math"], llm=gemini_llm)
tools.append(search_tool)

def general_tool(input_text):
    return gemini_llm.predict(input_text)

general_tool = Tool(
    name="general_task_solver",
    description="Handles tasks that do not require a specific tool.",
    func=general_tool
)

tools.append(general_tool)

# Create the ReAct agent
agent = create_react_agent(
    llm=gemini_llm,
    tools=tools,
    prompt=prompt
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5  # Add a maximum number of iterations to prevent infinite loops
)

# Run the agent
response = agent_executor.invoke({
    "input": "Write a an email to Jotman telling him to be serious with his Machine learning Engineering career"
})
print(response)

def check_creator_question(message):
    creator_keywords = ["who made you", "who created you", "who built you", "who developed you",
                       "your creator", "your developer", "who programmed you"]
def chatbot(message, history=None):
    if history is None:
        history = []

    # Handle creator-specific questions
    creator_keywords = ["who made you", "who created you", "who built you", "who developed you", 
                         "your creator", "your developer", "who programmed you"]
    if any(keyword in message.lower() for keyword in creator_keywords):
        response = "I was built by Adewuyi Ayomide, a Machine Learning Engineer and Computer Science student at University of Ibadan."
    else:
        try:
            # Use the agent for responses
            result = agent_executor.invoke({"input": message})
            response = result["output"]
        except Exception as e:
            response = f"I apologize, but I encountered an error: {str(e)}"

    # Update history
    history.append(("You", message))
    history.append(("AI", response))

    return history, history

# Create the Gradio interface using Blocks
with gr.Blocks() as interface:
    with gr.Row():
        chatbot_output = gr.Chatbot()
        state = gr.State([])  # Keep track of conversation history

    with gr.Row():
        with gr.Column(scale=10):
            input_text = gr.Textbox(placeholder="Type your message here...", label="Message", lines=1)
        with gr.Column(scale=2):
            send_button = gr.Button("Send")  # Custom "Send" button

    # Define the interaction
    send_button.click(fn=chatbot, inputs=[input_text, state], outputs=[chatbot_output, state])

if __name__ == "__main__":
    interface.launch(share=True)
