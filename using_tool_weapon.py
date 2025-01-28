import os, json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

openai = OpenAI(api_key=openai_api_key)
MODEL = 'gpt-4o-mini'

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

ticket_prices = {
    "london": "$799",
    "paris": "$899",
    "tokyo": "$1,400",
    "berlin": "$499",
    "seoul": "$1,599",
    "mars": "$2,250,000",
}


def get_ticket_price(destination):
    print(f"Tool get_ticket_price called for {destination}")
    city = destination.lower()
    return ticket_prices.get(city, "Unknown")


price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "The city or planet that the customer wants to travel to",
            },
        },
        "required": ["destination"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]


def chat(message, history):
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        print(message)
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    print(f"Tool call: {tool_call.json()}")
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination": city,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city


##########================================================================


gr.ChatInterface(fn=chat, type="messages").launch()
