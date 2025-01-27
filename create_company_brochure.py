

import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI
import gradio as gr


load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print(f"OpenAI API key is exists!!")
else:
    print("OpenAI API key is not exists!!")

MODEL = 'gpt-4o-mini'
openai = OpenAI()

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    url: str
    title: str
    body: str
    links: List[str]
    text: str

    def __init__(self, url: str):
        self.url = url
        response = requests.get(url, headers=headers)

        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else 'No title found'
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""

        links = [link.get('href') for link in soup.find_all('a') if link]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title: \n{self.title}\nWebpage Contents:\n{self.text}\n\n"

# few shot prompt
link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example. Do not include markdown formatting like triple backticks (```).:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""

def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt


def get_links(website):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)},
        ],
    )

    result = response.choices[0].message.content
    return json.loads(result)


#############################################

def get_all_details(url: str):
    result = "Landing page:\n"
    website = Website(url)
    result += website.get_contents()
    links = get_links(website)

    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result

brochure_system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."


def get_brochure_user_prompt(company_name, url: str):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:5_000]  # Truncate if more than 5,000 characters
    return user_prompt

def create_brochure(company_name, url: str, stream=False):

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)},
        ],
        stream=stream,
    )

    if stream:
        result = ""
        # display_handle = display(Markdown(""), display_id=True)

        for chunk in response:
            result += chunk.choices[0].delta.content or ''
            yield result
            # result = result.replace("```","").replace("markdown","")
            # update_display(Markdown(result), display_id=display_handle.display_id)
        # print(result)
    else:
        result = response.choices[0].message.content
        yield result
        # display(Markdown(result))
        # print(result)


# create_brochure("HuggingFace", "https://huggingface.co")
# print(create_brochure("Hippo's Logb Lab", "https://hippo-an.netlify.app"), True)


view = gr.Interface(
    fn=create_brochure,
    inputs=[
        gr.Textbox(label="Company name:"),
        gr.Textbox(label="Landing page URL including http:// or https://"),
        gr.Checkbox(label="Stream mode")],
    outputs=[gr.Markdown(label="Brochure:")],
    flagging_mode="never"
)
view.launch(share=True)
