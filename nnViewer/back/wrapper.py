import math
import warnings

import torch

from nnViewer.back.nodes import Node, AddNode, CatNode, ViewNode, GetItemNode, FonctionNode, MulNode, SubNode, PowNode, \
    MeanNode, ExpNode, DivNode, SumNode, MatMulNode, AttentionProductNode
from nnViewer.back.wrapped_function import function_to_wrap

wrapped_output = []
wrapped_nodes = []

def wrapper(function_name, function):
    def custom_function(*args, **kwargs):
        output = function(*args, **kwargs)
        if hasattr(output, "grad_fn"):
            if output.grad_fn is not None:
                visited = set()
                grad_fns_found = []

                def traverse_grad_fn(grad_fn):
                    if grad_fn in visited:
                        return
                    visited.add(grad_fn)

                    for arg in args:
                        if isinstance(arg, torch.Tensor) and arg.grad_fn == grad_fn:
                            return
                    for kwarg in kwargs.values():
                        if isinstance(kwarg, torch.Tensor) and kwarg.grad_fn == grad_fn:
                            return

                    grad_fns_found.append(str(id(grad_fn)))

                    for next_fn in grad_fn.next_functions:
                        if next_fn[0] is not None:
                            traverse_grad_fn(next_fn[0])

                traverse_grad_fn(output.grad_fn)

                wrapped_output.append({
                    "args": args,
                    "kwargs": kwargs,
                    "grad_fn_created":grad_fns_found[1:],
                    "node": create_fonction_node(output, args, kwargs, function_name, function)
                })

        return output
    return custom_function


def enable_function_wrapping():
    for original_function, module in function_to_wrap:
        function_name = original_function.__name__

        if not hasattr(module, f"_original_{function_name}"):
            setattr(module, f"_original_{function_name}", original_function)

        wrapped_function = wrapper(function_name, original_function)
        setattr(module, function_name, wrapped_function)

def unwrap_functions():
    for original_function, module in function_to_wrap:
        function_name = original_function.__name__

        if hasattr(module, f"_original_{function_name}"):
            original_function = getattr(module, f"_original_{function_name}")
            setattr(module, function_name, original_function)

def create_fonction_node(output, args, kwargs, function_name, function) -> Node:
    grad_fn_id = str(id(output.grad_fn))

    common_params = {
        "id": grad_fn_id,
        "name": function_name,
        "fonction": function,
    }

    try:
        if function_name in ["__add__", "__iadd__"]:
            return AddNode(mat1=args[0], mat2=args[1], output=output, **common_params)

        elif function_name in ["cat", "stack"]:
            return CatNode(input=args[0], output=output, **common_params)

        elif function_name in ["view", "transpose", "reshape", "expand", "flatten", "t", "permute"]:
            return ViewNode(input=args[0], output=output, **common_params)

        elif function_name in ["__mul__", "__rmul__"]:
            return MulNode(mat1=args[0], mat2=args[1], output=output, **common_params)

        elif function_name == "__sub__":
            return SubNode(mat1=args[0], mat2=args[1], output=output, **common_params)

        elif function_name == "__getitem__":
            if args[1]:
                if not any(slice is None for slice in args[1]):
                    slice_str = format_slice(args[1], args[0].shape)
                    return GetItemNode(input=args[0], slice=slice_str, output=output, **common_params)

            slice_str = f"{args[0].shape}->{output.shape}"
            return GetItemNode(input=args[0], slice=slice_str, output=output, **common_params)

        elif function_name == "pow":
            return PowNode(input=args[0], pow_value=args[1], output=output, **common_params)

        elif function_name == "__truediv__":
            return DivNode(mat1=args[0], mat2=args[1], output=output, **common_params)

        elif function_name == "exp":
            return ExpNode(exp_value=args[0], output=output, **common_params)

        elif function_name == "mean":
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            return MeanNode(input=args[0], output=output, dim=dim, **common_params)

        elif function_name == "sum":
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            return SumNode(input=args[0], output=output, dim=dim, **common_params)

        elif function_name in ["matmul", "__matmul__"]:
            return MatMulNode(mat1=args[0], mat2=args[1], output=output, **common_params)

        elif function_name == "scaled_dot_product_attention":
            mask = kwargs.get("mask", args[3] if len(args) > 3 else None)
            return AttentionProductNode(key=args[0],
                                        query=args[1],
                                        value=args[2],
                                        output=output,
                                        mask=mask,
                                        **common_params)

        else:
            warnings.warn(f"{function_name} not implemented")

    except:
        warnings.warn(f"Wrapped function {function_name} hasn't found its node")
        return FonctionNode(**common_params)

    return FonctionNode(**common_params)

def format_slice(slice_tuple, tensor_size):
    """
    Format a slice tuple into a readable string representation for PyTorch tensors.

    Args:
        slice_tuple: Tuple containing slice objects, integers, tensors, and/or Ellipsis
        tensor_size: torch.Size object representing tensor dimensions

    Returns:
        str: Formatted string representation of the slice
    """
    dimensions = len(tensor_size)
    slice_str = []

    for i, slice_item in enumerate(slice_tuple):
        # Handle tensors
        if isinstance(slice_item, torch.Tensor):
            tensor_list = slice_item.tolist()
            if isinstance(tensor_list, list):
                # Si tous les éléments sont consécutifs, utiliser le format start:stop
                if len(tensor_list) == 2 and tensor_list[1] - tensor_list[0] == 1:
                    slice_str.append(f"{tensor_list[0]}:{tensor_list[1]}")
                # Si les éléments sont identiques, utiliser une seule valeur
                elif all(x == tensor_list[0] for x in tensor_list):
                    slice_str.append(str(tensor_list[0]))
                else:
                    slice_str.append(str(tensor_list))
            else:
                slice_str.append(str(tensor_list))

        # Handle Ellipsis
        elif slice_item is Ellipsis:
            remaining_dims = dimensions - len(slice_tuple) + 1
            slice_str.extend([":"] * remaining_dims)

        # Handle slice objects
        elif isinstance(slice_item, slice):
            start, stop, step = slice_item.start, slice_item.stop, slice_item.step
            if start is None and stop is None and step is None:
                slice_str.append(":")
            else:
                slice_text = ""
                if start is not None:
                    slice_text += str(start)
                slice_text += ":"
                if stop is not None:
                    slice_text += str(stop)
                if step is not None:
                    slice_text += ":" + str(step)
                slice_str.append(slice_text)

        # Handle integer indices
        elif isinstance(slice_item, int):
            slice_str.append(str(slice_item))

        else:
            raise TypeError(f"Unsupported slice type: {type(slice_item)}")

    # Add missing dimensions as ":"
    while len(slice_str) < dimensions:
        slice_str.append(":")

    return f"[{', '.join(slice_str)}]"