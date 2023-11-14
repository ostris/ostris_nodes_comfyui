# modified from https://github.com/sayakpaul/caption-upsampling/blob/main/upsample_drawbench_captions.py
import os
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base_node import OstrisBaseNode
from ..settings.config import ostris_config
import folder_paths as comfy_paths


def make_final_message(
        system_message: Dict[str, str],
        rest_of_the_message: List[Dict[str, str]],
        debug=False,
):
    """Prepares the final message for inference."""
    final_message = [system_message]
    final_message.extend(rest_of_the_message)
    return final_message


def _upsample_caption(pipe, message):
    """Performs inference on a single prompt."""
    prompt = pipe.tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs


def _prepare_assistant_reply(assistant_output):
    """Prepares the assistant reply which will be considered as the upsampled caption."""
    output = assistant_output[0]["generated_text"]
    parts = output.rsplit("<|assistant|>", 1)
    assistant_reply = parts[1].strip() if len(parts) > 1 else None
    return assistant_reply


def _get_messages_for_chat() -> Tuple[Dict, List[Dict]]:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the DALL-E 3 technical report:
    https://cdn.openai.com/papers/dall-e-3.pdf.
    """
    system_message = {
        "role": "system",
        "content": """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

    There are a few rules to follow:

    - You will only ever output a single image description per user request.
    - Sometimes the user will request that you modify previous captions. In this case, you should refer to your previous conversations with the user and make the modifications requested.
    - When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.
    - Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user."
    - Image descriptions must be between 15-300 words. Extra words will be ignored.
    """,
    }
    # - Image descriptions must be between 15-80 words. Extra words will be ignored.

    rest_of_the_message = [
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input: 'a man holding a sword'",
        },
        {
            "role": "assistant",
            "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a blue light , casting a soft glow on the trees and bushes surrounding him.",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'make the light red'",
        },
        {
            "role": "assistant",
            "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a red light, casting a warm glow on the trees and bushes surrounding him.",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'draw a frog playing dominoes'",
        },
        {
            "role": "assistant",
            "content": "a frog sits on a worn table playing a game of dominoes with an elderly raccoon. the table is covered in a green cloth, and the frog is wearing a jacket and a pair of jeans. The scene is set in a forest, with a large tree in the background.",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : '{caption}'",
        },
    ]
    return system_message, rest_of_the_message


class OstrisLLMPipeLoader(OstrisBaseNode):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    "STRING", {
                        "default": 'HuggingFaceH4/zephyr-7b-beta',
                        "multiline": False
                    }
                ),
            }
        }

    RETURN_TYPES = ("LLM_PIPELINE",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load_llm_pipeline"

    CATEGORY = ostris_config.categories.llm

    def load_llm_pipeline(self, model_name):
        MODELS_DIR = comfy_paths.models_dir
        cache = os.path.join(MODELS_DIR, 'llm')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=False,
            revision="main",
            load_in_4bit=True,
            cache_dir=cache
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            # model_kwargs={
            #     'load_in_4bit': True
            # },
            # cache_dir=cache

        )
        return (pipe,)


class OstrisPromptUpsampler(OstrisBaseNode):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_pipe": ("LLM_PIPELINE",),
                "string": ("STRING", {"multiline": True}),
                "seed": ("SEED",),
            }
        }

    RETURN_TYPES = ("STRING", "TEXT")
    RETURN_NAMES = ("string", "text")
    FUNCTION = "upsample_prompt"

    CATEGORY = ostris_config.categories.llm

    def upsample_prompt(self, llm_pipe, string, seed):
        if llm_pipe is None:
            raise ValueError("Pipeline not loaded. Please call load_model() first.")

        seed_int = seed['seed']
        # save current seed
        # save current seed state for training
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_int)

        captions = [string]
        upsampled_captions = []
        for caption in captions:
            system_message, rest_of_the_message = _get_messages_for_chat()
            updated_prompt = rest_of_the_message[-1]["content"].format(caption=caption)
            rest_of_the_message[-1]["content"] = updated_prompt
            final_message = make_final_message(
                system_message, rest_of_the_message, debug=False
            )

            outputs = _upsample_caption(llm_pipe, final_message)
            upsampled_caption = _prepare_assistant_reply(outputs)
            upsampled_captions.append(upsampled_caption)

        # restore seed state
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        print(upsampled_captions)
        return (upsampled_captions[0], upsampled_captions[0],)


# For backwards compatibility
class OstrisCaptionUpsampler(OstrisPromptUpsampler):
    def __init__(self):
        super().__init__()
