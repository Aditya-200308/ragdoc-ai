import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, "src")
for d in [root_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import types
if "src" not in sys.modules:
    m = types.ModuleType("src")
    m.__path__ = [src_dir]
    sys.modules["src"] = m

import src.app
