"""Local (offline) ground for the infantry-rl tasks.

``sim_utils.spawn_ground_plane`` -- and therefore ``TerrainImporterCfg`` with
``terrain_type="plane"`` -- references a Nucleus-hosted USD and fails on this
machine, which has no internet access.  ``TerrainImporterCfg(terrain_type="usd")``
however just calls ``spawn_from_usd`` on a local file, which works offline.

This module ships that file (``ground.usda``: one ``UsdGeom.Plane`` carrying
``PhysicsCollisionAPI`` + ``PhysxCollisionAPI``) and rebuilds it if it is missing,
so the task never references an ``omniverse-content`` URL.
"""

from __future__ import annotations

from pathlib import Path

GROUND_USD_PATH: Path = Path(__file__).resolve().parent / "ground.usda"

_GROUND_USDA = """#usda 1.0
(
    defaultPrim = "ground"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "ground"
{
    def Plane "plane" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
    )
    {
        uniform token axis = "Z"
        float3 xformOp:scale = (200, 200, 1)
        uniform token[] xformOpOrder = ["xformOp:scale"]
        rel material:binding:physics = </ground/groundMaterial>
    }

    def Material "groundMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    )
    {
        float physics:dynamicFriction = 0.9
        float physics:staticFriction = 1.0
        float physics:restitution = 0.0
    }
}
"""


def ensure_ground_usd(path: Path | str | None = None) -> str:
    """Return the path to the local ground USD, writing it if it does not exist."""
    target = Path(path) if path is not None else GROUND_USD_PATH
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_GROUND_USDA, encoding="utf-8")
    return str(target)


__all__ = ["GROUND_USD_PATH", "ensure_ground_usd"]
