# run_train_with_ext.py
import os
import sys
import importlib
import importlib.util
from pathlib import Path

def _add_paths_and_patch_cli_args():
    this_file = Path(__file__).resolve()
    isaac_group_root = this_file.parents[3]  # .../ISAACLAB

    isaaclab_scripts = isaac_group_root / "IsaacLab" / "scripts"
    rsl_rl_dir = isaaclab_scripts / "reinforcement_learning" / "rsl_rl"

    # add paths in sys.path
    for p in (str(rsl_rl_dir), str(isaaclab_scripts)):
        if p not in sys.path:
            sys.path.insert(0, p)

    
    try:
        import cli_args  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        mod = importlib.import_module("scripts.reinforcement_learning.rsl_rl.cli_args")
        sys.modules["cli_args"] = mod
        return
    except Exception:
        pass

    cli_args_py = rsl_rl_dir / "cli_args.py"
    if not cli_args_py.exists():
        raise RuntimeError(f"not found cli_args.py with path: {cli_args_py}")
    spec = importlib.util.spec_from_file_location("cli_args", str(cli_args_py))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    sys.modules["cli_args"] = mod


def _prepare_env():
    os.environ["ISAACLAB_TASKS_EXTRA_PACKAGES"] = \
        "isaaclab_custom_ext,isaaclab_custom_ext.registration"


def _force_register_gym_ids():
    import importlib
    importlib.import_module("isaaclab_custom_ext.registration")


def main():
    _prepare_env()
    _add_paths_and_patch_cli_args()

    _force_register_gym_ids()

    from scripts.reinforcement_learning.rsl_rl import train
    train.main()


if __name__ == "__main__":
    main()

