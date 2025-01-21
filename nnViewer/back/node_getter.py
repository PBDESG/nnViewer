from typing import Dict, Union, Tuple, Set

from torch import Tensor

from nnViewer.back.nodes import VarNode, FonctionNode, Node
from nnViewer.back.utils import split_module_name

def get_node_from_base_tensor(var: Union[Tensor, Tuple[Tensor, ...]],
                              params: Dict,
                              stop_grad_fns: Set = None,
                              with_base_tensor: bool = True) -> Tuple[Set[Node], Set[Tuple[str, str]], bool]:
    stoped = False
    nodes = set()
    edges = set()

    if stop_grad_fns is None:
        stop_grad_fns = set()

    param_map = {id(v): k for k, v in params.items()} if params else {}
    seen = {None}

    def search_nodes(fn):
        nonlocal stoped
        if fn in seen or fn in stop_grad_fns:
            stoped = stoped or (fn in stop_grad_fns)
            return

        seen.add(fn)

        if hasattr(fn, 'variable'):
            var = fn.variable
            var_id = str(id(var))
            seen.add(var)
            name = split_module_name(param_map.get(id(var), "var"))[-1]
            nodes.add(VarNode(id=var_id, name=name, variable=var))
            edges.add((var_id, str(id(fn))))

        nodes.add(FonctionNode(id=str(id(fn)), name=str(type(fn).__name__), fonction=fn))

        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    edges.add((str(id(u[0])), str(id(fn))))
                    search_nodes(u[0])

    def add_base_tensor(var: Tensor):
        var_id = str(id(var))
        if var in seen:
            return

        seen.add(var)
        if with_base_tensor:
            nodes.add(VarNode(id=var_id, variable=var, name="output"))
        if var.grad_fn:
            search_nodes(var.grad_fn)
            if with_base_tensor:
                edges.add((str(id(var.grad_fn)), var_id))

        if var._is_view():
            add_base_tensor(var._base)
            if with_base_tensor:
                edges.add((str(id(var._base)), var_id))

    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    return nodes, edges, stoped
