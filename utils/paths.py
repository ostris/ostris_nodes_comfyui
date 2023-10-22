
def fix_import_paths():
    # does all the import magical stuff
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "ostris_nodes_comfyui"))
    import folder_paths as comfy_paths

    sys.path.append(comfy_paths.base_path)

