from .nodes.general import *

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
    }
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
