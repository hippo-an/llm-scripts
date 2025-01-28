import base64, os, json
from io import BytesIO
from openai import OpenAI
from PIL import Image
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=api_key)



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

MODEL = 'gpt-4o-mini'

def artist(city):
    print(f"Generate Image for {city}")

    image_response = openai.images.generate(
        model="dall-e-3",
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024x1024", n=1,
        response_format="b64_json",
    )

    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)

    return Image.open(BytesIO(image_data))


from pydub import AudioSegment
from pydub.playback import play


def talker(message):
    # tts
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",  # onyx, alloy
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)


def chat(history):
    print(f"Start chat with chatbot{history}")
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

    image = None

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        print(f"message: {message}")
        messages.append(response)
        print(f"response: {response}")
        image = artist(city)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]

    # talker(reply)

    return history, image


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


with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chatting with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Cleaarrrr")


    def do_entry(message, history):
        history += [{"role": "user", "content": message}]
        return "", history

    entry.submit(
        do_entry,
        inputs=[entry, chatbot],
        outputs=[entry, chatbot]
    ).then(
        chat,
        inputs=chatbot,
        outputs=[chatbot, image_output]
    )

    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)
