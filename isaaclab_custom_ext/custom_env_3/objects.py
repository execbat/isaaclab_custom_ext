from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCylinderCfg, MeshCuboidCfg	
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



OBSTACLE_CYL = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/obst",
    spawn=MeshCylinderCfg(
        radius=0.40,
        height=1.2,
        axis="Z",  # ось цилиндра; по умолчанию "Z"
        rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
        mass_props=MassPropertiesCfg(mass=1000.0),
        collision_props=CollisionPropertiesCfg(collision_enabled=True),
        visual_material=PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8), roughness=0.6),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.0)),
)
