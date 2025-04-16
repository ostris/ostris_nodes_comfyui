import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence, ImageFile
import folder_paths
from .base_node import OstrisBaseNode
from comfy.cli_args import args
from PIL.PngImagePlugin import PngInfo
from ..settings.config import ostris_config


def adain(content_features, style_features):
    # Assumes that the content and style features are of shape (batch_size, channels, width, height)

    dims = [2, 3]
    if len(content_features.shape) == 3:
        # content_features = content_features.unsqueeze(0)
        # style_features = style_features.unsqueeze(0)
        dims = [1, 2]

    # Step 1: Calculate mean and variance of content features
    content_mean, content_var = torch.mean(content_features, dim=dims, keepdim=True), torch.var(content_features,
                                                                                                dim=dims,
                                                                                                keepdim=True)
    # Step 2: Calculate mean and variance of style features
    style_mean, style_var = torch.mean(style_features, dim=dims, keepdim=True), torch.var(style_features, dim=dims,
                                                                                          keepdim=True)

    # Step 3: Normalize content features
    content_std = torch.sqrt(content_var + 1e-5)
    normalized_content = (content_features - content_mean) / content_std

    # Step 4: Scale and shift normalized content with style's statistics
    style_std = torch.sqrt(style_var + 1e-5)
    stylized_content = normalized_content * style_std + style_mean

    return stylized_content


class OstrisImagePadding(OstrisBaseNode):
    padding_methods = [
        'mirror',
        'replicate',
        'white',
        'black',
    ]

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (cls.padding_methods, {"default": cls.padding_methods[0]}),
                "left_padding": ("INT", {"default": 128, "min": 0, "max": 5000, "step": 1}),
                "right_padding": ("INT", {"default": 128, "min": 0, "max": 5000, "step": 1}),
                "top_padding": ("INT", {"default": 128, "min": 0, "max": 5000, "step": 1}),
                "bottom_padding": ("INT", {"default": 128, "min": 0, "max": 5000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pad_image"

    CATEGORY = ostris_config.categories.image

    def pad_image(
            self,
            image: torch.Tensor,
            method: str,
            left_padding: int,
            right_padding: int,
            top_padding: int,
            bottom_padding: int
    ):
        print('Input shape:', image.shape)
        # images are shape (bs, w, h, c)
        # convert to (bs, c, w, h)
        image = image.permute(0, 3, 1, 2)

        is_batch = len(image.shape) == 4
        if not is_batch:
            image = image.unsqueeze(0)

        kwargs = {}
        mode = 'constant'
        if method == 'mirror':
            mode = 'reflect'
        elif method == 'replicate':
            mode = 'replicate'
        elif method == 'white':
            mode = 'constant'
            kwargs['value'] = 1.0
        elif method == 'black':
            mode = 'constant'
            kwargs['value'] = 0.0

        # add padding
        image = torch.nn.functional.pad(
            image,
            (left_padding, right_padding, top_padding, bottom_padding),
            mode=mode,
            **kwargs
        )

        # convert back to (bs, w, h, c)
        image = image.permute(0, 2, 3, 1)

        # remove batch dim if not batch
        if not is_batch:
            image = image.squeeze(0)

        return (image,)


class OstrisImageAdain(OstrisBaseNode):

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content": ("IMAGE",),
                "style": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "do_it"

    CATEGORY = ostris_config.categories.image

    def do_it(
            self,
            content: torch.Tensor,
            style: torch.Tensor,
    ):
        print('content shape:', content.shape)
        print('style shape:', style.shape)

        is_batch = len(content.shape) == 4
        if not is_batch:
            image = content.unsqueeze(0)

        is_batch_style = len(style.shape) == 4
        if not is_batch_style:
            image = style.unsqueeze(0)

        # images are shape (bs, w, h, c)
        # convert to (bs, c, w, h)
        style = style.permute(0, 3, 1, 2)
        content = content.permute(0, 3, 1, 2)

        output = adain(content, style)

        # convert back to (bs, w, h, c)
        output = output.permute(0, 2, 3, 1)

        # remove batch dim if not batch
        if not is_batch:
            output = output.squeeze(0)

        return (output,)


class OstrisSaveImageDirect:
    def __init__(self):
        self.type = "output"
        self.output_dir = folder_paths.get_output_directory()
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The images to save."
                    }
                ),
                "caption": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "default": "",
                        "tooltip": "Caption to save with the image. Will be saved in a .txt file with the same name as the image."
                    }
                ),
                "save_path": (
                    "STRING",
                    {
                        "default": os.path.join(folder_paths.get_output_directory(), "output.png"),
                        "tooltip": "Full path to save image to."
                    }
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves an image to a specified path"

    def save_images(self, image, save_path, caption='', prompt=None, extra_pnginfo=None):
        results = list()
        if isinstance(image, list):
            image = image[0]
        if len(image.shape) == 4:
            image = image[0]

        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        file = os.path.basename(save_path)
        output_dir = os.path.dirname(save_path)
        os.makedirs(output_dir, exist_ok=True)
        if save_path.lower().endswith('.png'):
            img.save(save_path, pnginfo=metadata, compress_level=self.compress_level)
        elif save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
            img.save(save_path, quality=95, subsampling=0)
        elif save_path.lower().endswith('.webp'):
            img.save(save_path, quality=95)
        else:
            raise ValueError(f"Unsupported file type: {save_path}")
        if caption is not None and caption.strip() != '':
            with open(os.path.splitext(save_path)[0] + '.txt', 'w') as f:
                f.write(caption)

        self.output_dir = output_dir

        results.append({
            "filename": file,
            "subfolder": '',
            "type": self.type
        })

        return {
            "ui": {
                "images": results
            }
        }
