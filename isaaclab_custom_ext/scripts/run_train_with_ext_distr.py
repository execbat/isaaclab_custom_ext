# run_train_with_ext.py
import os
import sys
import importlib
import importlib.util
from pathlib import Path

PKG_NAME = "isaaclab_custom_ext"

def _locate_pkg_root() -> Path:
    """Find the root of the isaaclab_custom_ext package (works with both -m and pip -e)."""
    here = Path(__file__).resolve()
    # 1) поиск вверх по дереву
    for p in [here] + list(here.parents):
        if (p / "__init__.py").exists() and p.name == PKG_NAME:
            return p
        cand = p / PKG_NAME / "__init__.py"
        if cand.exists():
            return cand.parent
    # 2) если установлен (editable/regular)
    spec = importlib.util.find_spec(PKG_NAME)
    if spec and spec.origin:
        return Path(spec.origin).parent
    raise RuntimeError(f"Unable to find package {PKG_NAME}")

def _find_isaaclab_scripts_root() -> Path:
    """Find the IsaacLab/scripts directory, moving up from the current file."""
    here = Path(__file__).resolve()
    for p in here.parents:
        candidate = p / "IsaacLab" / "scripts"
        if candidate.exists():
            return candidate
    # as a fallback: if the IsaacLab module is already in PYTHONPATH
    try:
        mod = importlib.import_module("scripts")
        return Path(mod.__file__).resolve().parent
    except Exception as e:
        raise RuntimeError(
            "Couldn't find 'IsaacLab/scripts'. Run from the ISAACLAB repo "
            " or add it to your PYTHONPATH."
        ) from e

def _add_paths_and_patch_cli_args():
    isaaclab_scripts = _find_isaaclab_scripts_root()
    rsl_rl_dir = isaaclab_scripts / "reinforcement_learning" / "rsl_rl"

    for p in (str(rsl_rl_dir), str(isaaclab_scripts)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # put cli_args into the imported name 'cli_args'
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
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    sys.modules["cli_args"] = mod

def _prepare_env_and_paths():
    
    pkg_root = _locate_pkg_root()
    pkg_parent = str(pkg_root.parent)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)


    os.environ["ISAACLAB_TASKS_EXTRA_PACKAGES"] = (
        f"{PKG_NAME},{PKG_NAME}.registration"
    )

def _force_register_gym_ids():
    importlib.import_module(f"{PKG_NAME}.registration")

def main():
    _prepare_env_and_paths()
    _add_paths_and_patch_cli_args()
    _force_register_gym_ids()

    from scripts.reinforcement_learning.rsl_rl import train
    train.main()

if __name__ == "__main__":
    main()

