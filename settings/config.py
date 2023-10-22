import os
from ..info import INFO

OSTRIS_NODES_ROOT = os.path.dirname(os.path.dirname(__file__))
OSTRIS_STORAGE_FOLDER = os.path.join(OSTRIS_NODES_ROOT, 'storage')
OSTRIS_DB_FILE = os.path.join(OSTRIS_STORAGE_FOLDER, 'ostris.db.json')


class OstrisPaths:
    project_root = OSTRIS_NODES_ROOT
    storage_folder = OSTRIS_STORAGE_FOLDER
    db_file = OSTRIS_DB_FILE


class OstrisCategory:
    root = "ostris"
    general = f"{root}/general"
    text = f"{root}/text"
    image = f"{root}/image"
    batch = f"{root}/batch"


class OstrisConfig:
    version = INFO['version']
    categories = OstrisCategory
    paths = OstrisPaths


ostris_config = OstrisConfig()
