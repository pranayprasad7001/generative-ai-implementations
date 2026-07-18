from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

async def main():
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            # Note: Ensure the absolute path to the mathserver.py file is used if run from another directory
            "args": ["math-server.py"], 
            "transport": "stdio",
        },
        "weather": {
            # Note: FastMCP exposes the endpoint at "/mcp"
            "url": "http://localhost:8000/mcp", 
            "transport": "streamable-http",
        },
    })

    tools = await client.get_tools()
    llm = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key)
    agent = create_agent(llm, tools)

    # Note: The input format to follow MessagesState schema
    math_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What's (3+5) x 12?"}]
    })

    print("Math response:", math_response["messages"][-1].content)

    weather_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
    })

    print("Weather response:", weather_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())