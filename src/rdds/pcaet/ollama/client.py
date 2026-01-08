import ollama

OLLAMA_PORT: int = 11434
OLLAMA_HOST: str = 'ollama-server'
OLLAMA_ENDPOINT: str = f"{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_MODEL = 'llama3.2'


class Client(ollama.Client):

    def __init__(self, *args, **kwargs):
        kwargs.update({
            'host': OLLAMA_ENDPOINT,
        })
        super().__init__(*args, **kwargs)