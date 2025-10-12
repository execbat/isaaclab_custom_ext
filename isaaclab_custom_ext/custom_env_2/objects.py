from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import (
    CylinderCfg,
    RigidBodyPropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
    PreviewSurfaceCfg,
)

# === marker - target  ===
TARGET_MARKER = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Target",
    spawn=CylinderCfg(
        radius=0.08,
        height=0.02,
        rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
        mass_props=MassPropertiesCfg(mass=0.1),
        collision_props=CollisionPropertiesCfg(collision_enabled=True),

        visual_material=PreviewSurfaceCfg(diffuse_color=(0.05, 0.9, 0.2), roughness=0.2, metallic=0.0),
    ),
)


