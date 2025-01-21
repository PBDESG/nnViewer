from dataclasses import dataclass

@dataclass
class PosData:
    x:float = 0
    y:float = 0
    height: float = 0
    width: float = 0

@dataclass
class PosDataUpperModules:
    x:float = 0
    y:float = 0
    height: float = 0
    width: float = 0
    name: str = ""
    class_name: str = ""
    margin: float = 0
    level: float = 0
