from langchain.tools import tool
import html2text
import os
import functools

from rdds.pcaet.ollama import Client, OLLAMA_DEFAULT_OPTIONS, OLLAMA_MODEL

# wget https://grch37.ensembl.org/info/docs/tools/vep/script/vep_plugins.html
_VEP_PLUGIN_WEB_SOURCE = \
    TEST_REFERENCE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vep_plugins.html'))

# TODO: Caching

@tool
@functools.cache
def vep_plugin_metadata_tool(plugin_name: str) -> str:
    """
    For a VEP plugin name, return the description of the plugin and what the plugin infers.
    
    A VEP plugin is as sub field in the INFO part of the VCF file.

    :param plugin_name: A string
    :return: The description of the plugin as string
    """
    # Read cached VEP plugins webpage
    with open(_VEP_PLUGIN_WEB_SOURCE, 'r') as fp:
        html_data = fp.read()
    # Convert plugin information to Markdown format
    html_to_markdown_converter = html2text.HTML2Text()
    html_to_markdown_converter.bypass_tables = True
    markdown_data = html_to_markdown_converter.handle(html_data)
    client = Client()
    prompt = f"""
    Describe the following VEP plugin:
    \"{plugin_name}\"
    using the information provided in the following markdown document:\n
    """
    prompt += markdown_data
    msg = [
        {'role': 'user',
         'content': prompt}
    ]
    response = client.chat(messages=msg,
                           model=OLLAMA_MODEL,
                           options=OLLAMA_DEFAULT_OPTIONS)
    return response.message.content
