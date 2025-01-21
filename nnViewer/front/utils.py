import inspect
from typing import List, Tuple
import re

import numpy as np
from PyQt5.QtGui import QImage, QColor, QPixmap, qRgb
from PyQt5.QtWidgets import QGraphicsTextItem, QHBoxLayout, QLineEdit
from PyQt5.QtCore import Qt
from torch import Tensor
import torch

from nnViewer.back.nodes import ModuleNode, FonctionNode, VarNode, BMMNode, MulNode, AddNode, ViewNode, \
    ExpandNode, GetItemNode, Conv2dNode, EmbeddingNode, PowNode, MeanNode, StackNode, CatNode, Conv1dNode, ExpNode, \
    DivNode, SumNode, MatMulNode, AttentionProductNode, LinearNode


def get_tuple_of_tensors_shapes_as_string(tensor_tuple):
    return ", ".join(["None" if tensor is None else get_tensor_shape_as_string(tensor) for tensor in tensor_tuple])

def get_tensor_shape_as_string(tensor):
    return f"({', '.join(map(str, tensor.shape))})"

def split_module_name(name: str) -> List[str]:
    if name == ".":
        return [""]
    return re.split(r"\.+", name)

def get_node_info(node):
    def create_tensor_info(label, tensor, visualization_type="tensor"):
        return {
            "value": label,
            "tensor": tensor,
            "visualization_type": visualization_type
        }

    if isinstance(node, Conv2dNode):
        return {
            "Input Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.input)),
                node.input[0]
            ),
            "Output Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.output)),
                node.output[0]
            ),
            "Number of Parameters": "{:,}".format(node.nb_parameters),
            "Input Channels": str(node.module.in_channels),
            "Output Channels": str(node.module.out_channels),
            "Kernel Size": create_tensor_info(
                f"{str(node.module.kernel_size[0])} x {str(node.module.kernel_size[1])}",
                node.module.weight if hasattr(node.module, 'weight') else None,
                "kernel"
            ),
            "Stride": f"{str(node.module.stride[0])} x {str(node.module.stride[1])}",
            "Padding": f"{str(node.module.padding[0])} x {str(node.module.padding[1])}",
        }

    if isinstance(node, Conv1dNode):
        return {
            "Input Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.input)),
                node.input[0]
            ),
            "Output Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.output)),
                node.output[0]
            ),
            "Number of Parameters": "{:,}".format(node.nb_parameters),
            "Input Channels": str(node.module.in_channels),
            "Output Channels": str(node.module.out_channels),
            "Kernel Size": create_tensor_info(
                f"{str(node.module.kernel_size[0])}",
                node.module.weight,
                "kernel"
            ),
            "Stride": f"{str(node.module.stride[0])}",
            "Padding": f"{str(node.module.padding[0])}",
        }

    if isinstance(node, LinearNode):
        return {
            "Input Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.input)),
                node.input[0]
            ),
            "Output Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.output)),
                node.output[0]
            ),
            "Number of Parameters": "{:,}".format(node.nb_parameters),
            "Weights":create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.module.weight)),
                node.module.weight
            ),
            "Bias": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.module.bias)),
                node.module.bias
            )
        }

    if isinstance(node, EmbeddingNode):
        return {
            "Input Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.input)),
                node.input[0]
            ),
            "Output Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.output)),
                node.output[0]
            ),
            "Number of Parameters": "{:,}".format(node.nb_parameters),
            "Size of the Embedding Matrix": create_tensor_info(
                f"{str(node.module.num_embeddings)} x {str(node.module.embedding_dim)}",
                node.module.weight
            ),
            "Size of Embedding Vector": str(node.module.embedding_dim),
        }

    elif isinstance(node, ModuleNode):
        return {
            "Input Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.input)),
                node.input[0]
            ),
            "Output Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.output)),
                node.output[0]
            ),
            "Number of Parameters": "{:,}".format(node.nb_parameters),
        }

    elif isinstance(node, MulNode):
        return {
            "First Element": create_tensor_info(format_matrix_data(node.mat1), node.mat1),
            "Second Element": create_tensor_info(format_matrix_data(node.mat2), node.mat2),
            "Output": create_tensor_info(format_matrix_data(node.output), node.output),
        }

    elif isinstance(node, ExpNode):
        return {
            "Exponential value": str(node.exp_value),
            "Output": create_tensor_info(format_matrix_data(node.output), node.output),
        }

    elif isinstance(node, DivNode):
        return {
            "First Element": create_tensor_info(format_matrix_data(node.mat1), node.mat1),
            "Second Element": create_tensor_info(format_matrix_data(node.mat2), node.mat2),
            "Output": create_tensor_info(format_matrix_data(node.output), node.output),

        }

    elif isinstance(node, AddNode):
        return {
            "First Element": create_tensor_info(format_matrix_data(node.mat1), node.mat1),
            "Second Element": create_tensor_info(format_matrix_data(node.mat2), node.mat2),
            "Output": create_tensor_info(format_matrix_data(node.output), node.output),
        }

    elif isinstance(node, MatMulNode):
        return {
            "First Element": create_tensor_info(format_matrix_data(node.mat1), node.mat1),
            "Second Element": create_tensor_info(format_matrix_data(node.mat2), node.mat2),
            "Output": create_tensor_info(format_matrix_data(node.output), node.output),
        }

    elif isinstance(node, AttentionProductNode):
        return {
            "Key": create_tensor_info(format_matrix_data(node.key), node.key),
            "Query": create_tensor_info(format_matrix_data(node.query), node.query),
            "Value": create_tensor_info(format_matrix_data(node.value), node.value),
            "Attention Matrix": create_tensor_info(format_matrix_data(node.attention_matrix), node.attention_matrix),
            "mask": "Not masked attention" if not  node.mask else (format_matrix_data(node.mask), node.mask),
            "Output": create_tensor_info(format_matrix_data(node.output), node.output),
        }

    elif isinstance(node, PowNode):
        return {
            "Input Tensor Shape": create_tensor_info(format_matrix_data(node.input), node.input),
            "Pow Value": create_tensor_info(format_matrix_data(node.pow_value), node.pow_value),
        }

    elif isinstance(node, MeanNode):
        return {
            "Input Tensor Shape": create_tensor_info(format_matrix_data(node.input), node.input),
            "Output Tensor Shape": create_tensor_info(format_matrix_data(node.output), node.output),
            "Mean on Dim": str(node.dim)
        }

    elif isinstance(node, SumNode):
        return {
            "Input Tensor Shape": create_tensor_info(format_matrix_data(node.input), node.input),
            "Output Tensor Shape": create_tensor_info(format_matrix_data(node.output), node.output),
            "Sum on Dim": str(node.dim)
        }

    elif isinstance(node, CatNode):
        return {
            "Shape of Tensor to Stack": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.input)),
                node.input[0]
            ),
            "Output Tensor Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.output)),
                node.output[0]
            ),
        }

    elif isinstance(node, ViewNode):
        return {
            "Input Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.input)),
                node.input[0]
            ),
            "Output Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.output)),
                node.output[0]
            ),
        }

    elif isinstance(node, GetItemNode):
        return {
            "Input Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.input)),
                node.input[0]
            ),
            "Output Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.output)),
                node.output[0]
            ),
            "Slices": str(node.slice)
        }

    elif isinstance(node, ExpandNode):
        return {
            "Node Name": str(node.name),
        }

    elif isinstance(node, FonctionNode):
        return {
            "Node Name": str(node.name),
        }

    elif isinstance(node, VarNode):
        return {
            "Node Name": str(node.name),
            "Variable Shape": create_tensor_info(
                get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(node.variable)),
                node.variable
            ),
        }

def format_matrix_data(matrix):
    if isinstance(matrix, Tensor):
        return f"Tensor of shape: {get_tuple_of_tensors_shapes_as_string(get_var_as_tuple_tensor(matrix))}"
    return str(matrix)



def make_color_paler(color, factor=0.2):
    r, g, b = color

    new_r = int(r + (255 - r) * factor)
    new_g = int(g + (255 - g) * factor)
    new_b = int(b + (255 - b) * factor)

    return (
        min(255, max(0, new_r)),
        min(255, max(0, new_g)),
        min(255, max(0, new_b))
    )

def get_string(var):
    if isinstance(var, Tensor):
        return get_tensor_shape_as_string(var)
    elif isinstance(var, float):
        return str(var)
    elif isinstance(var, int):
        return str(var)
    elif isinstance(var, str):
        return var
    elif var is None:
        return "None"
    else:
        return str(var)

def get_var_as_tuple_tensor(var):
    if var is None:
        return (None)
    elif (not isinstance(var, Tensor)) and (not isinstance(var, Tuple)):
        var_output = []
        for _, value in var.__dict__.items():
            if isinstance(value, Tensor):
                var_output.append(value)
        return tuple(var_output)
    elif not isinstance(var, tuple):
        return (var,)
    else:
        return var

def create_centered_text_item(label_text, font):
    lines = label_text.splitlines()

    max_length = max(len(line) for line in lines)

    centered_lines = [line.center(max_length) for line in lines]

    centered_text = "\n".join(centered_lines)
    item = QGraphicsTextItem(centered_text)
    item.setFont(font)

    item.setTextWidth(item.boundingRect().width())

    item.setPos(-item.boundingRect().width() / 2, -item.boundingRect().height() / 2)

    return item


def create_image_from_matrix(matrix, color="gray"):
    height, width = matrix.shape

    # Calculate new dimensions while maintaining aspect ratio
    if height > 256 or width > 256:
        ratio = min(256 / height, 256 / width)
        new_height = int(height * ratio)
        new_width = int(width * ratio)

        # Resize the matrix using numpy
        from scipy.ndimage import zoom
        scale_factors = (new_height / height, new_width / width)
        matrix = zoom(matrix, scale_factors, order=0)
        height, width = new_height, new_width

    # Ensure matrix values are within [0, 255]
    matrix = np.clip(matrix, 0, 255).astype(np.uint8)

    # Create image with the new dimensions
    image = QImage(width, height, QImage.Format_RGB32)

    # Create the color array
    if color == "gray":
        for y in range(height):
            for x in range(width):
                val = int(matrix[y, x])
                image.setPixel(x, y, qRgb(val, val, val))
    else:  # red
        for y in range(height):
            for x in range(width):
                val = int(matrix[y, x])
                image.setPixel(x, y, qRgb(val, 0, 0))

    # Si l'image est plus petite que 256x256, on l'agrandit en mode pixelisé
    if width < 256 or height < 256:
        image = image.scaled(256, 256, Qt.KeepAspectRatio, Qt.FastTransformation)

    return QPixmap.fromImage(image)


def normalize_to_255(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    normalized_matrix = (matrix - min_val) / (max_val - min_val) * 255
    normalized_matrix = np.clip(normalized_matrix, 0, 255)  # Ensure values are in range [0, 255]

    return normalized_matrix.astype(np.uint8)

def get_image_from_slice_layout_and_tensor(slice_layout, tensor, color="gray"):
    slice = get_slice_from_layout(slice_layout)
    tensor = tensor.detach() if tensor.requires_grad else tensor
    slice_tensor = tensor[slice]

    slice_numpy = slice_tensor.numpy()

    slice_numpy = normalize_to_255(slice_numpy)

    return create_image_from_matrix(slice_numpy, color)

def get_slice_from_layout(slice_layout):
    slice_list = []
    for i in range(slice_layout.count()):
        s = slice_layout.itemAt(i).widget().text()
        if s == ':':
            slice_list.append(slice(None))
        else:
            slice_list.append(int(s))
    return tuple(slice_list)

def set_up_slice_layout_from_tensor(tensor):
    input_slice_layout = QHBoxLayout()
    slice_fields = []
    for i, _ in enumerate(tensor.shape):
        slice_field = QLineEdit()
        if i < len(tensor.shape) - 2:
            slice_field.setText("0")
        else:
            slice_field.setText(":")
        slice_field.setAlignment(Qt.AlignCenter)
        input_slice_layout.addWidget(slice_field)
        slice_field.setFixedWidth(18)
        slice_fields.append(slice_field)
    return input_slice_layout, slice_fields
