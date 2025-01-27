import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


if openai_api_key:
    print(f"OpenAI API key is exists!!")
else:
    print("OpenAI API key is not exists!!")


openai = OpenAI()
MODEL = 'gpt-4o-mini'
system_message = "You are a helpful assistant"

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""

    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

gr.ChatInterface(fn=chat, type="messages").launch()