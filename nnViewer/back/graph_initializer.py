import gc
from typing import Tuple, Callable, Dict, Optional
import torch

from torch import Tensor, nn

from nnViewer.back.graph import Graph
from nnViewer.back.node_getter import get_node_from_base_tensor
from nnViewer.back.utils import get_var_as_tuple_tensor
from nnViewer.back.hook_functions import hooks, root_node_belong_to_module, set_hooks, remove_hooks
from nnViewer.back.wrapper import unwrap_functions, wrapped_output, enable_function_wrapping

class GraphInitializer:
    def __init__(self,
                 model: nn.Module,
                 fn_to_wrap: Callable,):
        self.model = model
        if not fn_to_wrap:
            fn_to_wrap = model.forward
        self.forward_fn = fn_to_wrap
        self.graph = None
        setattr(self.model, fn_to_wrap.__name__, self.wrap_forward)

    def wrap_forward(self, *args, **kwargs):
        if self.graph:
            return self.forward_fn(*args, **kwargs)

        self.graph, output = build_nn_graph(self.model, self.forward_fn, args, kwargs)
        return output

def wrap_model(model, fn_to_wrap: Optional[Callable] = None) -> GraphInitializer:
    if isinstance(model, nn.Module):
        return GraphInitializer(model, fn_to_wrap)

    sub_models = (getattr(model, attr) for attr in dir(model))
    for sub_model in sub_models:
        if isinstance(sub_model, nn.Module):
            return GraphInitializer(sub_model, fn_to_wrap)

    raise ValueError("No nn.Module found")

def build_nn_graph(model: nn.Module,
                    forward_fn: Callable,
                    args: Tuple,
                    kwargs: Dict) -> Tuple[Graph, Tuple[Tensor]]:
    gc.disable()
    hook_handles = set_hooks(model)
    enable_function_wrapping()

    with torch.set_grad_enabled(True):
        output = forward_fn(*args, **kwargs)

    unwrap_functions()

    remove_hooks(hook_handles)

    output_tuple = get_var_as_tuple_tensor(output)
    output_tuple = tuple(output for output in output_tuple if hasattr(output, "grad_fn"))

    nodes, edges, _ = get_node_from_base_tensor(output_tuple, dict(model.named_parameters()))

    gc.enable()

    graph = Graph(nodes, edges)

    graph.set_up(hooks, root_node_belong_to_module,  wrapped_output)

    return graph, output
