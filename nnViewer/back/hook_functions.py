from typing import Callable, Tuple, Dict

import torch
from torch import nn, Tensor

from nnViewer.back.node_getter import get_node_from_base_tensor
from nnViewer.back.nodes import (LinearNode, LayerNormNode, ModuleNode, Conv2dNode,
                                 EmbeddingNode, Conv1dNode)
from nnViewer.back.utils import get_var_as_tuple_tensor

hooks = set()
known_nodes = set()
root_node_belong_to_module = {}


NODE_CLASSES = {
    "Linear": LinearNode,
    "LayerNorm": LayerNormNode,
    "Conv2d": Conv2dNode,
    "Conv1d": Conv1dNode,
    "Embedding": EmbeddingNode,
}

SKIP_MODULE = ["Dropout", "Identity"]

def hook_fn_factory(name: str) -> Callable:
    def hook_fn(module: nn.Module,
                args: Tuple,
                kwargs: Dict,
                output: Tuple) -> None:

        if not name or module.__class__.__name__ in SKIP_MODULE:
            return None

        input_tuple = tuple(
            arg for arg in list(args) + list(kwargs.values())
            if isinstance(arg, torch.Tensor)
        )

        module_id = create_unique_id()

        stop_grad_fns = {arg.grad_fn for arg in input_tuple
                         if hasattr(arg, "grad_fn") and arg.grad_fn is not None}

        output_tuple = get_var_as_tuple_tensor(output)
        nodes, _, _ = get_node_from_base_tensor(
            var = output_tuple,
            params = dict(module.named_parameters()),
            stop_grad_fns=stop_grad_fns,
            with_base_tensor=False
        )

        nodes_ids = {node.id for node in nodes}

        for id_root_node in nodes_ids:
            if id_root_node not in known_nodes:
                root_node_belong_to_module[id_root_node] = module_id
                known_nodes.add(id_root_node)

        if nodes_ids:
            node = create_module_node(
                module_id,
                name,
                input_tuple,
                module,
                output_tuple,
            )
            node.all_root_sub_ids = nodes_ids
            hooks.add(node)
    return hook_fn

def create_module_node(
                module_id: str,
                name:str,
                input_tuple: Tuple[Tensor],
                module: nn.Module,
                output_tuple:Tuple[Tensor]):
    node_class = NODE_CLASSES.get(module.__class__.__name__, ModuleNode)

    return node_class(
        module_id,
        name,
        input_tuple,
        module,
        output_tuple,
    )

def set_hooks(model):
    hook_handles = []
    if isinstance(model, nn.Module):
        for name, layer in model.named_modules():
            hook_handles.append(
                layer.register_forward_hook(hook_fn_factory(name), with_kwargs=True)
            )
    return hook_handles

def remove_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()

current_id = 0
def create_unique_id():
    global current_id
    current_id += 1
    return str(current_id)
