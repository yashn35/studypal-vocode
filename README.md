# studypal
### Have a conversation about any article on the web

studypal is a fast conversational ai built using:
- [Deepgram](https://deepgram.com/) for transcribing the user's speech
- [OpenAI](https://platform.openai.com/) `gpt-3.5-turbo-1106` as the LLM brain to generate responses
- [Cartesia](https://cartesia.ai)'s [Sonic](https://cartesia.ai/sonic) TTS model to generate human-like audio of the agent
- All of these are orchestrated together using [Vocode](https://www.vocode.dev/) and runs right in the command line

## Setup

1. Clone the repository
2. Create a `.env` file in the project root
3. Copy `.env.example` to `.env` and add API keys 
4. Install the required packages: `pip install -r requirements.txt` 
6. Now you are ready! Run `python3 studypal.py` from your command line. 