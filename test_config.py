from src.config import Lab2Config
import yaml

with open("configs/lab2_full.yaml", encoding="utf-8") as f:
    raw = yaml.safe_load(f)

cfg = Lab2Config(**raw)
print(cfg)
