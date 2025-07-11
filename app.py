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

