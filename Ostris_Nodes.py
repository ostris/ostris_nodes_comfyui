from .utils.paths import fix_import_paths
fix_import_paths()
from .nodes.general import OstrisOneSeedNode, OstrisTextBoxNode
from .nodes.llm import OstrisLLMPipeLoader, OstrisCaptionUpsampler
from .nodes.batch_image_loader import OstrisBatchImageLoader


ostris_node_list = [
    {
        "uid": "One Seed - Ostris",
        "class": OstrisOneSeedNode,
        "title": "One Seed",
    },
    {
        "uid": "Text Box - Ostris",
        "class": OstrisTextBoxNode,
        "title": "Text Box",
    },
    {
        "uid": "LLM Pipe Loader - Ostris",
        "class": OstrisLLMPipeLoader,
        "title": "LLM Pipe Loader",
    },
    {
        "uid": "LLM Caption Upsampling - Ostris",
        "class": OstrisCaptionUpsampler,
        "title": "Caption Upsampling",
    },
    # {
    #     "uid": "Batch Image Loader - Ostris",
    #     "class": OstrisBatchImageLoader,
    #     "title": "Text Box",
    # }
]

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {}

# build the mappings
for n in ostris_node_list:
    NODE_CLASS_MAPPINGS[n["uid"]] = n["class"]
    NODE_DISPLAY_NAME_MAPPINGS[n["uid"]] = f"{n['title']}_uid"
