from collections.abc import Generator
from typing import Any
from io import BytesIO

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from app import generate_slide_deck
from helpers.text_helper import get_clean_json


class OyGenPptxTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:

        llm_str = tool_parameters['llm_str']
        pptx_demo = tool_parameters['pptx_demo']
        json = get_clean_json(llm_str)
        if pptx_demo:
            ppt_demo_io = pptx_demo[0].blob
            result_file_bytes = generate_slide_deck(json, BytesIO(ppt_demo_io))
        else:
            result_file_bytes = generate_slide_deck(json, None)
        output_filename = tool_parameters['output_filename']

        if output_filename:
            result_filename = output_filename
        else:
            result_filename = 'Presentation'

        if not result_filename.endswith('.pptx'):
            result_filename += '.pptx'

        yield self.create_blob_message(
            blob=result_file_bytes.getvalue(),
            meta={
                "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "filename": result_filename,
            }
        )
