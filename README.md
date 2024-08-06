# studytool
### Have a conversation about any article on the web

studytool is a fast conversational ai you can talk to about any article on the internet. 

It is built using:
- [Deepgram](https://deepgram.com/) for transcription of what the user says
- `gpt-3.5-turbo-1106` from [OpenAI](https://platform.openai.com/) as the LLM brain to generate responses
- [Cartesia](https://cartesia.ai) [Sonic](https://cartesia.ai/sonic) TTS to generate audio of the agent
- All of these are orchestrated together using [Vocode](https://www.vocode.dev/) and runs right in the command line

## Setup

1. Clone the repository
2. Create a `.env` file in the project root
3. Copy `.env.example` to `.env` and add API keys 
4. Install the required packages: `pip install -r requirements.txt` 
6. Now you are ready! Run `python3 studytool.py` from your command line. 