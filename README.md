# oct-demo
an openai assistants api demo

## Smart function call

Add this function to OpenAI asssistant:

```json
{
  "name": "update_weather_forecast",
  "description": "Obtain the weather forecast for the next five days for a given city using the OpenWeatherMap API.",
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

```json
{
  "name": "update_weather",
  "description": "Obtain the current weather for a given location.",
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
