from .encoder import Encoder, OllamaClient, EncodingFormat, EncodingResult, OLLAMA_DEFAULT_OPTIONS
from .data_types import Document

class DocumentEncoder(Encoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._document_encoding_llm = OllamaClient()


    def _document_attribute_search(self, keyword: str, document_context: str) -> str:
        keyword = keyword.replace('_', ' ')  # Convert code attribute to human (LLM) readable format
        prompt = f"Here's a document:\n"
        prompt += "-"*10 + "\n"
        prompt += document_context
        prompt += "-"*10 + "\n"
        prompt += f"Return relevant items in the document related to this keyword: \"{keyword}\", as a list delimited by comma."
        prompt += "Don't return the keyword itself, be very concise and don't guess. Use information only present in the above document."
        msgs = [
            {'role': 'user',
             'content': prompt}
        ]
        response = self._document_encoding_llm.chat(model=self.ollama_model, messages=msgs, options=OLLAMA_DEFAULT_OPTIONS)
        return response.message.content

    def encode(self, document: Document) -> EncodingResult:
        """
        Encode Document to embedding representation
        :param document: A Document instance
        :return: EncodingResult
        """
        with open(document.path) as file:
            document_data = file.read()

        # Search the document for relevant attributes
        encoding_format = EncodingFormat()
        for key in vars(encoding_format):
            value = self._document_attribute_search(keyword=key, document_context=document_data)
            encoding_format.__setattr__(key, value)

        # Encode
        return super().encode(encoding_format)