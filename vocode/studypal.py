# Run this script directly from your command line. 

import asyncio
import signal
import requests
import os
import io
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.synthesizer.cartesia_synthesizer import CartesiaSynthesizer
from vocode.streaming.models.synthesizer import CartesiaSynthesizerConfig
from vocode.streaming.models.audio import AudioEncoding
from vocode.logging import configure_pretty_logging
import tiktoken

# Get API keys
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')
cartesia_api_key = os.getenv('CARTESIA_API_KEY')

def get_article_content(url):
    if 'arxiv.org' in url:
        return get_arxiv_content(url)
    else:
        return get_wikipedia_content(url)

# Helper function to extract content from Wikipedia url (this is technically agnostic to URL type but will work best with Wikipedia articles)
def get_wikipedia_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    content = soup.find('div', {'class': 'mw-parser-output'})
    
    if content:
        return content.get_text()
    else:
        return "Failed to extract Wikipedia article content."

# Helper function to extract content from arXiv url 
def get_arxiv_content(url):
    # Convert URL to PDF URL if necessary
    if '/abs/' in url:
        url = url.replace('/abs/', '/pdf/')
    if not url.endswith('.pdf'):
        url += '.pdf'

    response = requests.get(url)
    if response.status_code == 200:
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        return "Failed to download arXiv PDF."
    
# Count number of tokens used in model
def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Truncate the content
def truncate_content(content, max_tokens=14000):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(content)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return content    

# This writes the content locally to your machine so you can see what content is being passed to the agent as context
def save_content_to_file(content, filename="extracted_content.txt"):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

async def main():
    configure_pretty_logging() # Add this if you want to see debugging 

    url = input("Enter the URL of the article you would like to talk about: ")
    article_content = get_article_content(url)
    
    # Truncate content if article too long
    max_tokens =14000
    article_content = truncate_content(article_content, max_tokens)

    # Save article to file
    save_content_to_file(article_content)

    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(
        mic_sampling_rate=48000,
        speaker_sampling_rate=48000,
        use_default_devices=False,
    )

    # This uses Vocode to orchestrate everything together 
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=deepgram_api_key,
            ),
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=openai_api_key,
                initial_message=BaseMessage(text="Hello! I'm ready to discuss the article with you. What would you like to learn about?"),
                prompt_preamble=f"""You are an AI study partner. You have been given the following article content:

{article_content}  

Your task is to help the user concisely understand and learn from this article. THESE RESPONSES SHOULD BE ONLY 1-3 SENTENCES AND CONCISE. THIS INSTRUCTION IS VERY IMPORTANT. RESPONSES SHOULDN'T BE LONG.""",
            )
        ),
        synthesizer=CartesiaSynthesizer(
            CartesiaSynthesizerConfig.from_output_device(
                speaker_output,
                api_key=cartesia_api_key,
                model_id='sonic-english',
                voice_id='79a125e8-cd45-4c13-8a67-188112f4dd22',
            )
        ),
    )
    
    await conversation.start()
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))
    
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)

if __name__ == "__main__":
    asyncio.run(main())