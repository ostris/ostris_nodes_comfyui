# modified from https://github.com/sayakpaul/caption-upsampling/blob/main/upsample_drawbench_captions.py
import os
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

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
    # if the assistant also provided user side input, remove it
    if assistant_reply and "<|user|>" in assistant_reply:
        assistant_reply = assistant_reply.split("<|user|>")[0].strip()
    if assistant_reply and "<|system|>" in assistant_reply:
        assistant_reply = assistant_reply.split("<|system|>")[0].strip()

    # remove any newlines
    assistant_reply = assistant_reply.replace("\n", " ").strip()
    return assistant_reply


def _get_messages_for_chat() -> Tuple[Dict, List[Dict]]:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the DALL-E 3 technical report:
    https://cdn.openai.com/papers/dall-e-3.pdf.
    """
    system_message = {
        "role": "system",
        "content": """\
You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say. \
For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" \
will trigger your partner bot to output an image of a forest morning, as described. \
You be given simple prompts that you will upsample to create, amazing images. \
Upsampling a prompt this is to take the short prompts and make them extremely detailed and descriptive while \
maintaining all of the information from the original prompt.

There are a few rules to follow:

- You will only ever output a single image description per user request.
- You will never use the same image description twice. You must always create very unique image descriptions.
- You will ALWAYS include all of the information from the input prompt, but you will add more details to it.
- Image descriptions must be between 15-300 words. Extra words will be ignored.
- You must be creative and imaginative with your image descriptions. They must be unique and interesting.
""",
    }

    rest_of_the_message = [
        {
            "role": "user",
            "content": "Upsample this prompt: 'a man holding a sword'",
        },
        {
            "role": "assistant",
            "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head, the blade glows with a blue light, casting a soft glow on the trees and bushes surrounding him",
        },
        {
            "role": "user",
            "content": "Upsample this prompt: 'frog playing dominoes'",
        },
        {
            "role": "assistant",
            "content": "a green tree frog sits on a worn table playing a game of dominoes, across the table is an elderly raccoon smoking a cigar, the table is covered in a green cloth and drinks, and the frog is wearing a jacket and a pair of jeans, the scene is set in a forest, with a large tree in the background",
        },
        {
            "role": "user",
            "content": "Upsample this prompt: '{prompt}'",
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

        trust_remote_code = False
        if model_name.startswith("stabilityai"):
            trust_remote_code = True

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            revision="main",
            cache_dir=cache,
            quantization_config=quantization_config
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=cache)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # guess at chat template if there is none. This works for stablelm
        if not tokenizer.chat_template:
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}<|user|>{{ message['content'] }}{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}{% elif message['role'] == 'system' %}<|system|>{{ message['content'] }}{% endif %}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}<|assistant|>"
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

        prompts = [string]
        upsampled_captions = []
        for prompt in prompts:
            system_message, rest_of_the_message = _get_messages_for_chat()
            updated_prompt = rest_of_the_message[-1]["content"].format(prompt=prompt)
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
