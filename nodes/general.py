from .base_node import OstrisBaseNode
from ..settings.config import ostris_config


class OstrisOneSeedNode(OstrisBaseNode):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":
                {
                    "seed": (
                        "INT", {
                            "default": 0,
                            "min": 0,
                            "max": 0xffffffffffffffff
                        }
                    )
                }
        }

    RETURN_TYPES = (
        "SEED",
        "INT",
        "NUMBER",
        "FLOAT",
        "STRING",
        "STRING"
    )

    RETURN_NAMES = (
        "seed",
        "int",
        "number",
        "float",
        "string",
        "zfill"
    )

    FUNCTION = "get_seed"

    CATEGORY = ostris_config.categories.general

    def get_seed(self, seed):
        max_length = 16
        seed = str(seed)
        if len(seed) > max_length:
            seed = seed[:max_length]
        return (
            {"seed": seed},
            int(seed),
            int(seed),
            float(seed),
            str(seed),
            str(seed).zfill(max_length)
        )


class OstrisTextBoxNode(OstrisBaseNode):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING", {
                        "default": '',
                        "multiline": True
                    }
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_multiline"

    CATEGORY = ostris_config.categories.text

    def text_multiline(self, text):
        import io
        new_text = []
        for line in io.StringIO(text):
            if not line.strip().startswith('#'):
                if not line.strip().startswith("\n"):
                    line = line.replace("\n", '')
                new_text.append(line)
        new_text = "\n".join(new_text)
        return (new_text,)
