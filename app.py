# """
# Streamlit app containing the UI and the application logic.
# """
import logging
import os
from io import BytesIO
from typing import Any

import json5
# import streamlit as st
from dotenv import load_dotenv

from global_config import GlobalConfig
from helpers import text_helper, pptx_helper

load_dotenv()

RUN_IN_OFFLINE_MODE = os.getenv('RUN_IN_OFFLINE_MODE', 'False').lower() == 'true'


def handle_error(error_msg: str, should_log: bool):
    """
    Display an error message in the app.

    :param error_msg: The error message to be displayed.
    :param should_log: If `True`, log the message.
    """

    if should_log:
        logger.error(error_msg)


logger = logging.getLogger(__name__)


def generate_slide_deck(json_str: str, ppt_demo_io: BytesIO | None) -> BytesIO | None | Any:
    """
    Create a slide deck and return the file path. In case there is any error creating the slide
    deck, the path may be to an empty file.

    :param json_str: The content in *valid* JSON format.
    :return: The path to the .pptx file or `None` in case of error.
    """
    blob = None
    try:
        parsed_data = json5.loads(json_str)
    except ValueError:
        handle_error(
            'Encountered error while parsing JSON...will fix it and retry',
            True
        )
        try:
            parsed_data = json5.loads(text_helper.fix_malformed_json(json_str))
        except ValueError:
            handle_error(
                'Encountered an error again while fixing JSON...'
                'the slide deck cannot be created, unfortunately ☹'
                '\nPlease try again later.',
                True
            )
            return None
    except RecursionError:
        handle_error(
            'Encountered a recursion error while parsing JSON...'
            'the slide deck cannot be created, unfortunately ☹'
            '\nPlease try again later.',
            True
        )
        return None
    except Exception:
        handle_error(
            'Encountered an error while parsing JSON...'
            'the slide deck cannot be created, unfortunately ☹'
            '\nPlease try again later.',
            True
        )
        return None


    try:
        logger.debug('Creating PPTX file blob')
        headers, blob = pptx_helper.generate_powerpoint_presentation(
            parsed_data,
            ppt_demo_io
        )
    except Exception as ex:
        # st.error(APP_TEXT['content_generation_error'])
        logger.error('Caught a generic exception: %s', str(ex))

    return blob

def generate_pptx(template_name, pptx_json_str,output_file_name):
     # 假设你已经有一个二进制的 PPTX 文件内容
    with open(f"./pptx_templates/{template_name}.pptx", "rb") as f:
        pptx_bytes = f.read()

    # 使用 BytesIO 在内存中加载
    pptx_stream = BytesIO(pptx_bytes)
    
    pptx_content = pptx_json_str
    
    blob = generate_slide_deck(json_str=pptx_content, ppt_demo_io=pptx_stream)
    
    if blob is None:
        logger.error("Failed to generate slide deck; no output was produced.")
    else:
        # Support either BytesIO or raw bytes
        out_path = f"./ppt/{output_file_name}.pptx"
        try:
            if isinstance(blob, BytesIO):
                blob.seek(0)
                with open(out_path, "wb") as out_f:
                    out_f.write(blob.read())
            elif isinstance(blob, (bytes, bytearray)):
                with open(out_path, "wb") as out_f:
                    out_f.write(blob)
            else:
                # Try to coerce via buffer interface
                with open(out_path, "wb") as out_f:
                    out_f.write(memoryview(blob).tobytes())
            logger.info("Wrote slide deck to %s", out_path)
        except Exception as ex:
            logger.exception("Could not write PPTX to disk: %s", ex)
    


# if __name__ == '__main__':
#     # 假设你已经有一个二进制的 PPTX 文件内容
#     with open("./pptx_templates/changlong.pptx", "rb") as f:
#         pptx_bytes = f.read()

#     # 使用 BytesIO 在内存中加载
#     pptx_stream = BytesIO(pptx_bytes)
    
#     pptx_content = """```json
#                     {
#                         "title": "Introduction to Python Programming",
#                         "slides": [
#                             {
#                                 "heading": "Slide 1: Introduction",
#                                 "bullet_points": [
#                                     "Brief overview of Python and its importance",
#                                     "Purpose of the tutorial"
#                                 ]
#                             },
#                             {
#                                 "heading": "Slide 2: Basic Data Types",
#                                 "bullet_points": [
#                                     "Strings (e.g. \"hello\")",
#                                     "Integers (e.g. 42)",
#                                     "Floats (e.g. 3.14)",
#                                     "Booleans (e.g. True/False)",
#                                     "Lists (e.g. [1, 2, 3])",
#                                     "Tuples (e.g. (1, 2, 3))"
#                                 ]
#                             },
#                             {
#                                 "heading": "Slide 3: Strings",
#                                 "bullet_points": [
#                                     "String literals (e.g. \"hello\")",
#                                     "String concatenation (e.g. \"hello\" + \" world\")",
#                                     "String slicing (e.g. \"hello\"[0] = h)"
#                                 ]
#                             },
#                             {
#                                 "heading": "Slide 4: Integers",
#                                 "bullet_points": [
#                                     "Integer literals (e.g. 42)",
#                                     "Arithmetic operations (e.g. 2 + 3 = 5)"
#                                 ]
#                             },
#                             {
#                                 "heading": "Slide 5: Floats",
#                                 "bullet_points": [
#                                     "Floating-point literals (e.g."
#                                 ]
#                             }
#                         ]
#                     }
#                     ```
#                     """
    
#     blob = generate_slide_deck(json_str=pptx_content, ppt_demo_io=pptx_stream)
    
#     if blob is None:
#         logger.error("Failed to generate slide deck; no output was produced.")
#     else:
#         # Support either BytesIO or raw bytes
#         out_path = "./ppt/default.pptx"
#         try:
#             if isinstance(blob, BytesIO):
#                 blob.seek(0)
#                 with open(out_path, "wb") as out_f:
#                     out_f.write(blob.read())
#             elif isinstance(blob, (bytes, bytearray)):
#                 with open(out_path, "wb") as out_f:
#                     out_f.write(blob)
#             else:
#                 # Try to coerce via buffer interface
#                 with open(out_path, "wb") as out_f:
#                     out_f.write(memoryview(blob).tobytes())
#             logger.info("Wrote slide deck to %s", out_path)
#         except Exception as ex:
#             logger.exception("Could not write PPTX to disk: %s", ex)