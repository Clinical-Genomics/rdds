from dataclasses import dataclass, asdict as dataclass_as_dict
import numpy as np
from json import dumps

from rdds.pcaet.ollama import Client as OllamaClient, OLLAMA_DEFAULT_OPTIONS
from .encoding_format import EncodingFormat

# TODO: Increase Ollama model context size

# https://reference.langchain.com/python/integrations/langchain_ollama/

@dataclass
class EncodingResult:
    input: EncodingFormat
    encoding: np.ndarray


class Encoder:

    def __init__(self):
        self._ollama_model = 'llama3.2'
        self._encoding_model_llm = OllamaClient()

    @property
    def ollama_model(self) -> str:
        return self._ollama_model

    def encode(self, encoding_format: EncodingFormat) -> EncodingResult:
        """
        Encode data in encoding_format to embeddings.
        :param encoding_format:
        :return:
        """
        json_data = dumps(dataclass_as_dict(encoding_format))
        # TODO: Investigate impact of lowercase vs non-modified input data on performance
        json_data = json_data.lower()  # Force lowercase of all input data (is this simplifying data or data degradation)?
        embed_response = self._encoding_model_llm.embed(model=self.ollama_model, input=json_data, options=OLLAMA_DEFAULT_OPTIONS)
        assert len(embed_response.embeddings) == 1, 'Got more than 1 expected embedding for input data'
        embeddings = embed_response.embeddings[0]
        embeddings = np.asarray(embeddings)
        encoding_result = EncodingResult(input=encoding_format, encoding=embeddings)
        return encoding_result
