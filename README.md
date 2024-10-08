# oct-demo
an openai assistants api demo

## Smart function call

Add this function to OpenAI asssistant:

```json
{
  "name": "update_weather_forecast",
  "description": "Obtain the three-hour interval weather forecast for the next five days for a given city using the OpenWeatherMap API. This forecast provides weather data at three-hour intervals for a total of 40 data points per day, covering a five-day period.",
  "strict": false,
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and full state, e.g., Portland, Oregon."
      }
    },
    "required": [
      "location"
    ]
  }
}

```
