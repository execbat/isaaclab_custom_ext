import os, sys, importlib, importlib.util
from pathlib import Path

def _force_register_gym_ids():
    os.environ.setdefault("ISAACLAB_TASKS_EXTRA_PACKAGES", "isaaclab_custom_ext")
    importlib.import_module("isaaclab_custom_ext.registration")

def _add_paths_and_patch_cli_args():
    this_file = Path(__file__).resolve()
    isaac_root = this_file.parents[3]
    isaaclab_scripts = isaac_root / "IsaacLab" / "scripts"
    rsl_rl_dir = isaaclab_scripts / "reinforcement_learning" / "rsl_rl"
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
    spec = importlib.util.spec_from_file_location("cli_args", str(rsl_rl_dir / "cli_args.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    sys.modules["cli_args"] = mod

def main():
    _force_register_gym_ids()
    _add_paths_and_patch_cli_args()
    from scripts.reinforcement_learning.rsl_rl import play
    play.main()

if __name__ == "__main__":
    main()

