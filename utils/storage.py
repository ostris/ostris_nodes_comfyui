import json
import os
import typing
from collections import OrderedDict

from ..settings.config import ostris_config

base_db = OrderedDict({
    'version': ostris_config.version,
    'nodes': {},
})


class OstrisDB:
    _db: OrderedDict

    def __init__(self):
        self.db_path = ostris_config.paths.db_file
        # make sure paths exist
        if not os.path.exists(ostris_config.paths.storage_folder):
            os.makedirs(ostris_config.paths.storage_folder, exist_ok=True)
        if not os.path.exists(ostris_config.paths.db_file):
            with open(ostris_config.paths.db_file, 'w') as f:
                f.write(json.dumps(base_db, indent=4))

        self._load_db()
        self._migrate_db()

    def _save_db(self):
        with open(self.db_path, 'w') as f:
            f.write(json.dumps(self._db, indent=4))

    def _migrate_db(self):
        data = self._db
        did_update = False
        for key, value in base_db.items():
            if key not in data:
                data[key] = value
                did_update = True
            if key == 'version':
                data[key] = value
                did_update = True
        self._db = data
        if did_update:
            self._save_db()
        return data, did_update

    def _load_db(self):
        with open(self.db_path, 'r') as f:
            self._db = json.loads(f.read(), object_pairs_hook=OrderedDict)

    def save_node_data(self, node_id, key, value):
        if node_id not in self._db:
            self._db[node_id] = {}
        self._db[node_id][key] = value
        self._save_db()

    def get_node_data(self, node_id, key, default=None):
        if node_id not in self._db:
            return default
        if key not in self._db[node_id]:
            return default
        return self._db[node_id][key]


# only keep one instance of this to keep them all in sync
_shared_db = None

# prevent recursive import
if typing.TYPE_CHECKING:
    from ..nodes.base_node import OstrisBaseNode


class OstrisNodeStorage:
    db: 'OstrisDB'

    def __init__(
            self,
            node_class_name: str,
    ):
        self._node_class_name_ = node_class_name

        global _shared_db
        if _shared_db is None:
            _shared_db = OstrisDB()
        self._db = _shared_db

    def save(self, key: str, value):
        self._db.save_node_data(self._node_class_name_, key, value)

    def get(self, key: str, default=None):
        return self._db.get_node_data(self._node_class_name_, key, default)
