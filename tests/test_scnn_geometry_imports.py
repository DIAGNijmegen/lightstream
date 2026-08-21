import ast
from pathlib import Path


def test_scnn_imports_shared_tile_geometry_helpers():
    scnn_path = Path(__file__).parents[1] / "lightstream/core/scnn/scnn.py"
    module = ast.parse(scnn_path.read_text())

    geometry_imports = {
        alias.name
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "lightstream.core.engine.geometry"
        for alias in node.names
    }

    assert {"tile_grid", "iter_tiles"} <= geometry_imports
