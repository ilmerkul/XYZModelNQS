from dataclasses import asdict
from pathlib import Path

from src.model.struct import ChainConfig

prj_root = Path(__file__).parent.parent.parent.absolute()


def report_name(cfg: ChainConfig) -> str:
    report_file_name = "report_"
    report_file_name += cfg.name + "_"
    report_file_name += "_".join(
        map(lambda kv: f"{kv[0]}({kv[1]})", asdict(cfg).items())
    )
    report_file_name += ".csv"
    return report_file_name


def report_path(cfg: ChainConfig, path_data: str) -> Path:
    report_file_name = report_name(cfg)
    report_file = prj_root.joinpath(path_data).joinpath(report_file_name)
    return report_file
