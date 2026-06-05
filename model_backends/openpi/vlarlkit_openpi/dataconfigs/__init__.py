from vlarlkit_openpi.dataconfigs.libero_dataconfig import LeRobotLiberoDataConfig
from vlarlkit_openpi.dataconfigs.miniskill_dataconfig import LeRobotManiSkillDataConfig
from vlarlkit_openpi.dataconfigs.robotwin_dataconfig import LeRobotRobotwinDataConfig


DATA_CONFIGS = {
    "libero": LeRobotLiberoDataConfig,
    "maniskill": LeRobotManiSkillDataConfig,
    "robotwin": LeRobotRobotwinDataConfig,
}


def get_data_config(name: str):
    if name not in DATA_CONFIGS:
        raise ValueError(f"Data config {name} not found")
    return DATA_CONFIGS[name]
