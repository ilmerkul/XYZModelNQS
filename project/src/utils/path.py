import hashlib
from pathlib import Path

from omegaconf import OmegaConf

prj_root = Path(__file__).parent.parent.parent.absolute()


def report_name(cfg) -> str:
    yaml_str = OmegaConf.to_yaml(cfg)
    report_file_name = hashlib.md5(yaml_str.encode("utf-8"))
    return report_file_name.hexdigest()


def report_path(cfg, path_data: str) -> Path:
    report_file_name = report_name(cfg)
    report_file = prj_root.joinpath(path_data).joinpath(report_file_name + ".csv")
    return report_file


def report_name_with_data(cfg, data: str) -> str:
    yaml_str = OmegaConf.to_yaml(cfg) + data
    report_file_name = hashlib.md5(yaml_str.encode("utf-8"))
    return report_file_name.hexdigest()
