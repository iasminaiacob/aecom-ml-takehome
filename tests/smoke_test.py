from __future__ import annotations

from src.utils.config import load_yaml
from src.utils.device import get_device
from src.utils.seed import seed_everything

def main() -> None:
    cfg = load_yaml("configs/train.yaml")
    seed = int(cfg["train"]["seed"])
    deterministic = bool(cfg["train"]["deterministic"])
    seed_everything(seed, deterministic=deterministic)

    device = get_device(cfg["device"]["prefer"])
    print("Config loaded OK.")
    print("Device:", device)

if __name__ == "__main__":
    main()
