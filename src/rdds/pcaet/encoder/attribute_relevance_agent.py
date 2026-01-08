from langchain.agents import create_agent
import functools
from langchain_ollama import ChatOllama
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel
from dataclasses import dataclass

from rdds.pcaet.ollama import OLLAMA_MODEL, OLLAMA_ENDPOINT

from .vep_plugin_metadata_tool import vep_plugin_metadata_tool

class IsRelevant(BaseModel):
    """ Whether the VCF field was relevant or not"""
    is_relevant: bool  # Whether the result was relevant or not

class VariantAttributeRelevanceAgent:

    def __init__(self):
        pass

    @functools.cache
    def infer_relevance_of_variant_field_to_format_keyword(self,
                                                           variant_vcf_info_attribute: str,
                                                           encoding_format_keyword: str):
        print(f"CHECKING {variant_vcf_info_attribute} -> {encoding_format_keyword}")
        model = ChatOllama(model=OLLAMA_MODEL, base_url=f"http://{OLLAMA_ENDPOINT}")
        agent = create_agent(model=model,
                             tools=[vep_plugin_metadata_tool],
                             #response_format=ToolStrategy(IsRelevant),
                             #response_format=ToolStrategy(IsRelevant, handle_errors=False),
                             debug=True)
        content = f"""
        Can this VCF field \"{variant_vcf_info_attribute}\"
        contain information relevant to inferring {encoding_format_keyword}?
        """
        system_prompt = """
        You're tasked with finding information related to VCF fields and plugins.
        A VCF is a file format containing information related to a patient genetic variation.
        """
        msg = {"messages": [
            {"role": "system",
             "content": system_prompt},
            {"role": "user",
             "content": content}
        ]}
        result = agent.invoke(msg)
        print(result)
        mined_results = result["messages"][-1].content
        return mined_results