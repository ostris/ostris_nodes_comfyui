# Ostris Nodes ComfyUI

This is a collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 
that I made for some QOL. I will be adding much more
advanced ones in the future once I get more familiar with the API.

## Installation
Just git clone this repo into your ComfyUI `custom_nodes` folder and restart.
```bash
cd <your_comfyui_folder>/custom_nodes
git clone https://github.com/ostris/ostris_nodes_comfyui.git
cd ostris_nodes_comfyui
pip install --upgrade -r requirements.txt
```


## Current Nodes

### General
 - **One Seed** - A universal seed node with numerous output formats
    - **seed** (SEED)
    - **int** (int)
    - **number** (NUMBER)
    - **float** (FLOAT)
    - **string** (STRING)
    - **zfill** (STRING) - zero filled to 16 digits
 - **Text Box** - Just a simple textbox for now
    - **string** (STRING)
    - **text** (TEXT)
 
### LLM Caption Upsampling (BETA)
Based on the fantastic work of [sayakpaul/caption-upsampling](
https://github.com/sayakpaul/caption-upsampling). It uses
an LLM to expand your prompt into a more complex prompt with more descriptive detail.

**WARNING:** This is experimental. It will likely remove TI embeddings. It loads in 4bit
mode but will still be VRAM hungry. 

- **LLM Pipe Loader** - Loads the LLM pipeline to use
  - **model_name** (STRING) Huggingface model name e.g. `HuggingFaceH4/zephyr-7b-alpha`
- **LLM Caption Upsampling** - Upsamples a prompt


 Example:
<img src="https://raw.githubusercontent.com/ostris/ostris_nodes_comfyui/main/assets/prompt_upsampling_demo.jpg" width="768" height="auto"> 
