import os, sys, importlib

def _ensure_orbit_in_path():
    orbit_root = os.environ.get("ORBIT_ROOT", os.getcwd())
    if orbit_root not in sys.path:
        sys.path.insert(0, orbit_root)
    if not os.path.exists(os.path.join(orbit_root, "cli_args.py")):
        raise RuntimeError(f"cli_args.py не найден по ORBIT_ROOT={orbit_root}")
    return orbit_root

def _register_gym_ids():
    import isaac_hydra_ext  # noqa: F401
    # форсируем модуль, где вызывается gym.register(...)
    try:
        importlib.import_module(
            "isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg"
        )
    except ModuleNotFoundError:
        pass

def main():
    _register_gym_ids()
    _ensure_orbit_in_path()
    from scripts.reinforcement_learning.rsl_rl import train  # импорт ТОЛЬКО здесь
    train.main()

if __name__ == "__main__":
    main()
