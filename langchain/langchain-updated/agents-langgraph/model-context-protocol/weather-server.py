from mcp.server.fastmcp import FastMCP

weather = FastMCP("weather")

@weather.tool(name="get_weather", description="Get the weather for a city")
async def get_weather(city: str) -> str:
    """Get the weather for a city"""
    return f"The weather in {city} is sunny"


if __name__ == "__main__":
    weather.run(transport="streamable-http")


# transport="http": 
# This is useful for short-running requests where the response is a single JSON-RPC message

# transport="streamable-http": 
# The response is streamed to the client as it is generated, rather than waiting for the entire response to be generated before sending it