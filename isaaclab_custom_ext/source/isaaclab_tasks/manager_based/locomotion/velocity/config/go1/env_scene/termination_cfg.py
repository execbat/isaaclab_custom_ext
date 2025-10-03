from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class ChaseTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )
    
#    root_too_low = DoneTerm(
#        func=mdp.root_height_below_minimum,
#        params={
#            "minimum_height": 0.20,                 # порог по высоте, м
#            "asset_cfg": SceneEntityCfg("robot"), 
#        },
#    )
