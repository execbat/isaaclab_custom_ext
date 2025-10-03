import os
import sys
import importlib
import importlib.util
from pathlib import Path


def _register_gym_ids():
    # import your extension where gym.register(...) happens
    import isaaclab_custom_ext  # noqa: F401
    try:
        importlib.import_module("isaaclab_custom_ext.rough_env_cfg")
    except ModuleNotFoundError:
        pass


def _add_paths_and_patch_cli_args():
    """
    Ensures that `import cli_args` inside train.py works without changing the IsaacLab source code.
    """
    this_file = Path(__file__).resolve()
    isaac_group_root = this_file.parents[3]

    isaaclab_scripts = isaac_group_root / "IsaacLab" / "scripts"
    rsl_rl_dir = isaaclab_scripts / "reinforcement_learning" / "rsl_rl"

    # Add paths to sys.path (to the beginning)
    for p in (str(rsl_rl_dir), str(isaaclab_scripts)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Let's try a regular import of cli_args
    try:
        import cli_args  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    # Let's try absolute import and register it under the name "cli_args"
    try:
        mod = importlib.import_module("scripts.reinforcement_learning.rsl_rl.cli_args")
        sys.modules["cli_args"] = mod
        return
    except Exception:
        pass

    # upload directly from the file and register
    cli_args_py = rsl_rl_dir / "cli_args.py"
    if not cli_args_py.exists():
        raise RuntimeError(
            f"Cli_args.py not found at path: {cli_args_py}\n"
            f"Check location IsaacLab/scripts/reinforcement_learning/rsl_rl."
        )
    spec = importlib.util.spec_from_file_location("cli_args", str(cli_args_py))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to create spec for cli_args.py"
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    sys.modules["cli_args"] = mod


def main():
    _register_gym_ids()
    _add_paths_and_patch_cli_args()

    from scripts.reinforcement_learning.rsl_rl import train  
    train.main()


if __name__ == "__main__":
    main()

