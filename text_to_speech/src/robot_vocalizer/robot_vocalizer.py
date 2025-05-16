import logging
import subprocess
import tempfile

from deepgram import (
    DeepgramClient,
    SpeakOptions,
)

logger = logging.getLogger(__name__)


class RobotVocalizer:
    def __init__(self, api_key: str, model: str = "aura-2-thalia-en"):
        self.deepgram = DeepgramClient(api_key)
        self.model = model
        self.options = SpeakOptions(model=model)

    def vocalize(self, text: str):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            response = self.deepgram.speak.rest.v("1").save(
                tmp_file.name, {"text": text}, self.options
            )
            logger.debug(f"Vocalized response: {response.to_json(indent=4)}")
            # subprocess.run(["afplay", tmp_file.name]) # Mac
            subprocess.run(["ffplay", tmp_file.name])
