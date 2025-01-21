from typing import List, Tuple, Dict
import re

from torch import Tensor

from nnViewer.back.models import PosData, PosDataUpperModules


def parse_pos(pos_str):
    points = pos_str.replace('e', '').replace(',', ' ').split()
    points = [(float(points[i]), -float(points[i+1])) for i in range(0, len(points), 2)]
    last_point = points.pop(0)
    points.append(last_point)
    return points

def split_module_name(name: str) -> List[str]:
    if name == ".":
        return [""]
    return re.split(r"\.+", name)

def create_bounding_rectangle(rectangles: List[PosData],
                              class_name: str,
                              margin_height: float = 15,
                              margin_width: float = 15,
                              level: float = 0):
    min_x = rectangles[0].x - rectangles[0].width / 2
    min_y = rectangles[0].y - rectangles[0].height / 2
    max_x = rectangles[0].x + rectangles[0].width / 2
    max_y = rectangles[0].y + rectangles[0].height / 2

    for rect in rectangles[1:]:
        min_x = min(min_x, rect.x - rect.width / 2)
        min_y = min(min_y, rect.y - rect.height / 2)
        max_x = max(max_x, rect.x + rect.width / 2)
        max_y = max(max_y, rect.y + rect.height / 2)

    bounding_width = max_x - min_x
    bounding_height = max_y - min_y

    return PosDataUpperModules(
        height=bounding_height + margin_height*2,
        width=bounding_width + margin_width*2,
        margin = margin_width,
        x=(min_x + max_x)/2,
        y=(min_y + max_y)/2,
        class_name=class_name,
        level=level
    )


def get_var_as_tuple_tensor(var):
    var_output = []
    if isinstance(var, tuple):
        for l in var:
            if isinstance(l, Tensor):
                var_output.append(l)
        return tuple(var_output)

    elif isinstance(var, Tensor):
        return (var,)

    elif isinstance(var, Dict):
        for _, value in var.items():
            if isinstance(value, Tensor):
                var_output.append(value)
        return tuple(var_output)

    elif (not isinstance(var, Tensor)) and (not isinstance(var, Tuple)):
        for _, value in var.__dict__.items():
            if isinstance(value, Tensor):
                var_output.append(value)
        return tuple(var_output)

    else:
        return var
