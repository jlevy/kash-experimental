from pathlib import Path

from kash.exec import import_action_subdirs

# This hook can be used for auto-registering actions from any module.
import_action_subdirs(["actions"], __package__, Path(__file__).parent)


import kash.kits.research.libs.query.query_commands  # noqa: F401
import kash.kits.research.libs.viz.viz_commands  # noqa: F401
