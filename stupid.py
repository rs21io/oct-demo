from llama_index.core import Document, SummaryIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.wikipedia import WikipediaReader
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from tavily import TavilyClient  # need to experiment with this some more for sure.
import requests
from llama_index.core.agent import (
    FunctionCallingAgentWorker,
    ReActAgent,
)

import os
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgentWorker
import gradio as gr

# from llama_index.core.tools.ondemand_loader_tool import(
#    OnDemandLoaderTool)

load_dotenv()

# Initialize your LLM settings (adjust as needed)
llm = OpenAI(model="gpt-4o-mini", temperature=0.3)
embed_model = OpenAIEmbedding()
Settings.llm = llm
Settings.embed_model = embed_model

# Need ReAct agent    ok
# https://www.youtube.com/watch?v=ZzPaHgiB3kk


# llm = Ollama(model="llama3", temperature=0, request_timeout=1_000)
# nemotronmini = Ollama(model="nemotron-mini", temperature=0, request_timeout=1_000)
Settings.llm = llm


def web_search(query: str) -> str:
    """
    Performs a web search using the Tavily API and returns the context string.

    Parameters:
    - query (str): The search query.

    Returns:
    - str: The context string from the Tavily API or an error message.
    """
    try:
        # Step 1: Instantiate the TavilyClient
        tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # Step 2: Execute the search query
        context = tavily_client.get_search_context(query=query)

        # Step 3: Return the context
        return f"**Web Search Context:**\n{context}"
    except Exception as e:
        return f"Error performing web search: {str(e)}"


def update_weather(location: str) -> str:
    api_key = os.environ["OPENWEATHERMAP_API_KEY"]
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": "imperial"}
    response = requests.get(base_url, params=params)
    weather_data = response.json()

    if response.status_code != 200:
        return f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}"

    lon = weather_data["coord"]["lon"]
    lat = weather_data["coord"]["lat"]
    main = weather_data["weather"][0]["main"]
    feels_like = weather_data["main"]["feels_like"]
    temp_min = weather_data["main"]["temp_min"]
    temp_max = weather_data["main"]["temp_max"]
    pressure = weather_data["main"]["pressure"]
    visibility = weather_data["visibility"]
    wind_speed = weather_data["wind"]["speed"]
    wind_deg = weather_data["wind"]["deg"]
    sunrise = datetime.fromtimestamp(weather_data["sys"]["sunrise"]).strftime(
        "%H:%M:%S"
    )
    sunset = datetime.fromtimestamp(weather_data["sys"]["sunset"]).strftime("%H:%M:%S")
    temp = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]
    condition = weather_data["weather"][0]["description"]

    return f"""**Weather in {location}:**
- **Coordinates:** (lon: {lon}, lat: {lat})
- **Temperature:** {temp:.2f}°F (Feels like: {feels_like:.2f}°F)
- **Min Temperature:** {temp_min:.2f}°F, **Max Temperature:** {temp_max:.2f}°F
- **Humidity:** {humidity}%
- **Condition:** {condition.capitalize()}
- **Pressure:** {pressure} hPa
- **Visibility:** {visibility} meters
- **Wind Speed:** {wind_speed} m/s, **Wind Direction:** {wind_deg}°
- **Sunrise:** {sunrise}, **Sunset:** {sunset}"""


def update_weather_forecast(location: str) -> str:
    """Fetches the weather forecast for a given location and returns a formatted string
    Parameters:
    - location: the search term to find weather information
    Returns:
    A formatted string containing the weather forecast data
    """

    api_key = os.environ["OPENWEATHERMAP_API_KEY"]
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": location,
        "appid": api_key,
        "units": "imperial",
        "cnt": 40,  # Request 40 data points (5 days * 8 three-hour periods)
    }
    response = requests.get(base_url, params=params)
    weather_data = response.json()
    if response.status_code != 200:
        return f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}"

    # Organize forecast data per date
    forecast_data = {}
    for item in weather_data["list"]:
        dt_txt = item["dt_txt"]  # 'YYYY-MM-DD HH:MM:SS'
        date_str = dt_txt.split(" ")[0]  # 'YYYY-MM-DD'
        time_str = dt_txt.split(" ")[1]  # 'HH:MM:SS'
        forecast_data.setdefault(date_str, [])
        forecast_data[date_str].append(
            {
                "time": time_str,
                "temp": item["main"]["temp"],
                "feels_like": item["main"]["feels_like"],
                "humidity": item["main"]["humidity"],
                "pressure": item["main"]["pressure"],
                "wind_speed": item["wind"]["speed"],
                "wind_deg": item["wind"]["deg"],
                "condition": item["weather"][0]["description"],
                "visibility": item.get(
                    "visibility", "N/A"
                ),  # sometimes visibility may be missing
            }
        )

    # Process data to create daily summaries
    daily_summaries = {}
    for date_str, forecasts in forecast_data.items():
        temps = [f["temp"] for f in forecasts]
        feels_likes = [f["feels_like"] for f in forecasts]
        humidities = [f["humidity"] for f in forecasts]
        pressures = [f["pressure"] for f in forecasts]
        wind_speeds = [f["wind_speed"] for f in forecasts]
        conditions = [f["condition"] for f in forecasts]

        min_temp = min(temps)
        max_temp = max(temps)
        avg_temp = sum(temps) / len(temps)
        avg_feels_like = sum(feels_likes) / len(feels_likes)
        avg_humidity = sum(humidities) / len(humidities)
        avg_pressure = sum(pressures) / len(pressures)
        avg_wind_speed = sum(wind_speeds) / len(wind_speeds)

        # Find the most common weather condition
        condition_counts = Counter(conditions)
        most_common_condition = condition_counts.most_common(1)[0][0]

        daily_summaries[date_str] = {
            "min_temp": min_temp,
            "max_temp": max_temp,
            "avg_temp": avg_temp,
            "avg_feels_like": avg_feels_like,
            "avg_humidity": avg_humidity,
            "avg_pressure": avg_pressure,
            "avg_wind_speed": avg_wind_speed,
            "condition": most_common_condition,
        }

    # Build the formatted string
    city_name = weather_data["city"]["name"]
    ret_str = f"**5-Day Weather Forecast for {city_name}:**\n"

    for date_str in sorted(daily_summaries.keys()):
        summary = daily_summaries[date_str]
        ret_str += f"\n**{date_str}:**\n"
        ret_str += f"- **Condition:** {summary['condition'].capitalize()}\n"
        ret_str += f"- **Min Temperature:** {summary['min_temp']:.2f}°F\n"
        ret_str += f"- **Max Temperature:** {summary['max_temp']:.2f}°F\n"
        ret_str += f"- **Average Temperature:** {summary['avg_temp']:.2f}°F (Feels like {summary['avg_feels_like']:.2f}°F)\n"
        ret_str += f"- **Humidity:** {summary['avg_humidity']:.0f}%\n"
        ret_str += f"- **Pressure:** {summary['avg_pressure']:.0f} hPa\n"
        ret_str += f"- **Wind Speed:** {summary['avg_wind_speed']:.2f} m/s\n"

    return ret_str


web_search_tool = FunctionTool.from_defaults(fn=web_search)
update_weather_tool = FunctionTool.from_defaults(fn=update_weather)
update_weather_forecast_tool = FunctionTool.from_defaults(fn=update_weather_forecast)

# Load documents and create index
loader = WikipediaReader()
documents = loader.load_data(
    pages=["Molchat Doma", "Bauhaus", "Building Automation Systems"]
)
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)
index = SummaryIndex(nodes)
wikipedia_query_engine = index.as_query_engine(streaming=True)

query_engine_tools_and_web_search = [
    QueryEngineTool(
        query_engine=wikipedia_query_engine,
        metadata=ToolMetadata(
            name="Molchat_Doma_Bauhaus_Building_Automation_Systems",
            description="""Provides information about Molchat Doma, Bauhaus, And Building
            Automation Systems from Wikipedia sources""",
        ),
    ),
    web_search_tool,
    update_weather_tool,
    update_weather_forecast_tool,
]

# I think FunctionCallingAgentWorker is the one to use for sure. not react.

agent_worker = FunctionCallingAgentWorker.from_tools(
    query_engine_tools_and_web_search,
    llm=Settings.llm,
    verbose=True,
    allow_parallel_tool_calls=True,
)
agent = agent_worker.as_agent()

react_agent = ReActAgent.from_tools(
    query_engine_tools_and_web_search, llm=Settings.llm, verbose=True
)


# Modify the chat_function to handle streaming
def chat_function(user_input, history):
    response = query_engine.query(user_input)
    partial_response = ""
    for token in response.response_gen:
        partial_response += token
        # Yield the updated history with the partial response
        # yield history + [(user_input, partial_response)]
        yield partial_response
    # At the end, ensure the final response is included


# yield history + [(user_input, partial_response)]


# need to add in history; nah this agent chat already has memory
# Define the Gradio ChatInterface
def original_chat_function(user_input, history):
    response = agent.chat(user_input)  # or agent
    return str(response)


# Create the Gradio interface with streaming enabled
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Ask me anything about Molchat Doma!")
    chatbot = gr.ChatInterface(
        fn=original_chat_function,
        chatbot=gr.Chatbot(
            height=550, show_copy_button=True, show_copy_all_button=True
        ),
        #  stream=True  # Enable streaming in the interface
    )

# Launch the app
demo.launch(share=True)
