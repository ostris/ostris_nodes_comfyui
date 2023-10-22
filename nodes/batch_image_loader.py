import hashlib
import os

import numpy as np
import torch

from .base_node import OstrisBaseNode
from ..settings.config import ostris_config
from glob import glob
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont

from ..utils.storage import OstrisNodeStorage


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


OstrisBatchImageLoaderStorage = OstrisNodeStorage('OstrisBatchImageLoader')


class ImgProcess:
    to_process_image_path_list = []
    processed_image_path_list = []
    failed_image_path_list = []


OstrisBatchImgProcess = ImgProcess()


class OstrisBatchImageLoader(OstrisBaseNode):
    last_folder_path = ''
    img_process = OstrisBatchImgProcess

    def __init__(self):
        super().__init__()
        self.storage = OstrisBatchImageLoaderStorage

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": (
                    "STRING", {
                        "default": OstrisBatchImageLoaderStorage.get('last_folder_path', ''),
                        "multiline": False
                    }
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
        "STRING",
        "STRING"
        "STRING"
        "STRING"
    )
    RETURN_NAMES = (
        "image",
        "filename",
        "filename_no_ext",
        "folder_path",
        "full_path",
        "caption"
    )
    FUNCTION = "batch_from_folder"

    CATEGORY = f"{ostris_config.categories.batch}"

    def load_image_paths(self, folder_path):
        img_exts = ['jpg', 'jpeg', 'png', 'webp', 'tiff']
        img_paths = []
        files = glob(os.path.join(folder_path, '*'))
        for file in files:
            if file.split('.')[-1].lower() in img_exts:
                img_paths.append(file)
        self.img_process.to_process_image_path_list = img_paths
        self.img_process.processed_image_path_list = []
        self.img_process.failed_image_path_list = []

    def get_caption(self, image_path):
        # caption files will be named the same as the image without ext, but with a .txt or a .caption extension
        caption_exts = ['txt', 'caption']
        path_no_ext = os.path.splitext(image_path)[0]
        possible_caption_paths = [f"{path_no_ext}.{ext}" for ext in caption_exts]

        caption_path = None

        for path in possible_caption_paths:
            if os.path.exists(path):
                caption_path = path
                break

        if caption_path is None:
            return None
        else:
            with open(caption_path, 'r') as f:
                raw = f.read()
                arr = raw.split('\n')
                clean = ", ".join([x.strip() for x in arr])
                return clean

    def batch_from_folder(self, **kwargs):
        print('batch_from_folder', kwargs)
        path = kwargs['folder_path']
        self.storage.save('last_folder_path', path)

        if not os.path.exists(path):
            print('path does not exist')
            raise Exception('Path does not exist')

        if len(self.img_process.to_process_image_path_list) == 0:
            print('No queued images, loading from folder')
            # build our image paths list
            self.load_image_paths(path)

        image_path = self.img_process.to_process_image_path_list.pop(0)
        try:
            # process image
            image = Image.open(image_path)
            # flip based on meta
            image = ImageOps.exif_transpose(image)
            output_image = pil2tensor(image)
            output_folder_path = os.path.dirname(image_path)
            output_full_path = image_path
            output_filename = os.path.basename(image_path)
            output_filename_no_ext = os.path.splitext(output_filename)[0]
            output_caption = self.get_caption(image_path)

        except Exception as e:
            print('OstrisBatchImageLoader ERROR:', e)
            self.img_process.failed_image_path_list.append(image_path)
            # try next image
            return self.batch_from_folder(**kwargs)

        self.img_process.processed_image_path_list.append(image_path)

        return (
            output_image,
            output_filename,
            output_filename_no_ext,
            output_folder_path,
            output_full_path,
            output_caption
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        print('IS_CHANGED', kwargs)
        print('cls.img_process', cls.img_process)
        if len(cls.img_process.to_process_image_path_list):
            return cls.img_process.to_process_image_path_list[0]
        return False
