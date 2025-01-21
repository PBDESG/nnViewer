import warnings
from copy import copy
from typing import Tuple, Dict, List, Union, Set

from pygraphviz import AGraph
from torch import Tensor
import torch

from nnViewer.back.nodes import Node, VarNode, ModuleNode, FonctionNode
from nnViewer.back.utils import create_bounding_rectangle, parse_pos

SAVED_PREFIX = "_saved_"
NODE_FUNCTION_TO_DELETE = ["AccumulateGrad", "ToCopyBackward0", "CloneBackward0", "SliceBackward0"]
ACCUMULATE_GRAD_FN = ["AccumulateGrad"]

class Graph():
    def __init__(self,
                 nodes: Set[Node],
                 edges: Set[Tuple[str, str]]):

        self.nodes = set()
        self.edges = set()
        self.ids = set()

        self.add_nodes(nodes)
        self.add_edges(edges)

        self.module_nodes = set()
        self.flying_nodes = set()
        self.flying_edges = set()
        self.flying_upper_modules = []
        self.max_number_parameters = 0
        self.level_max = 0
        self.pos_edges = []

    def add_nodes(self, nodes: Union[Node, Set[Node]]) -> None:
        if isinstance(nodes, Node):
            nodes = {nodes}
        self.nodes.update(nodes)
        self.ids.update({node.id for node in nodes})

    def set_level_max(self) -> None:
        for node in self.nodes:
            node_level = len(node.up_modules) - 1
            self.level_max = max(self.level_max, node_level)

    def get_node(self, node_id: str, graph_name:str="all") -> Union[Node, bool]:
        graph_map = {
            "all": self.nodes,
            "flying": self.flying_nodes,
            "module": self.module_nodes
        }

        nodes = graph_map.get(graph_name)
        if nodes is None:
            raise Exception(f"Unknown graph name: {graph_name}")

        return next((node for node in nodes if node.id == node_id), False)

    def get_nodes(self, node_ids: List[str]) -> List[Node]:
        return [self.get_node(idx) for idx in node_ids]

    def get_module_class_name(self) -> List[str]:
        return list(set(sorted([node.module.__class__.__name__ for node in self.module_nodes])))

    def delete_nodes_type(self, funcs_to_delete: List[str]) -> None:
        for node in copy(self.nodes):
            if isinstance(node, FonctionNode):
                if node.fonction.__class__.__name__ in funcs_to_delete:
                    self.remove_node_and_reset_relatives(node)
        self.reset_edges()

    def safe_delete(self,
                    node:Node) -> None:
        for child in node.childrens:
            if node in child.parents:
                child.parents.remove(node)

        for parent_node in node.parents:
            if node in parent_node.childrens:
                parent_node.childrens.remove(node)

        if node in self.nodes:
            self.nodes.remove(node)

        if node.id in self.ids:
            self.ids.remove(node.id)

        for module_node in self.module_nodes:
            if node.id in module_node.all_root_sub_ids:
                module_node.all_root_sub_ids.remove(node.id)
            if node in module_node.sub_nodes:
                module_node.sub_nodes.remove(node)

    def remove_node_and_reset_relatives(self, node: Node) -> None:
        childrens = node.childrens
        parents = node.parents
        for child in childrens:
            if node in child.parents:
                child.parents.remove(node)
            child.parents.extend(parents)
        for parent in parents:
            if node in parent.childrens:
                parent.childrens.remove(node)
            parent.childrens.extend(childrens)
        self.nodes.remove(node)
        self.ids.remove(node.id)

        for module_node in self.module_nodes:
            if node.id in module_node.all_root_sub_ids:
                module_node.all_root_sub_ids.remove(node.id)
            if node in module_node.sub_nodes:
                module_node.sub_nodes.remove(node)

    def add_edge(self,
                 edge: Tuple[str, str]) -> None:
        self.edges.add(edge)

    def add_edges(self, edges: Union[Tuple[str, str], Set[Tuple[str, str]]]) -> None:
        if isinstance(edges, tuple):
            edges = {edges}
        self.edges.update(edges)

    def set_relatives(self)-> None:
        for edge in self.edges:
            head_node = self.get_node(edge[1])
            tail_node = self.get_node(edge[0])
            head_node.add_parent(tail_node)
            tail_node.add_children(head_node)

    def set_sub_parents_and_childrens(self)-> None:
        for node in self.module_nodes:
            for sub_node_id in node.all_root_sub_ids:
                sub_node = self.get_node(sub_node_id)

                node.all_sub_childrens.extend(sub_node.next_ids)
                node.all_sub_parents.extend(sub_node.previous_ids)

    def set_next_data(self) -> None:
        for node in self.nodes:
            node.next_ids = [child.id for child in node.childrens]

    def set_previous_data(self) -> None:
        for node in self.nodes:
            node.previous_ids = [parent.id for parent in node.parents]

    def init_flying_graph(self) -> None :
        for flying_node in self.nodes:
            if not flying_node.up_modules:
                self.set_flying_relatives(flying_node)
                self.flying_nodes.add(flying_node)

        self.set_flying_edges()

    def set_flying_edges(self) -> None:
        self.flying_edges = set()
        for node in self.flying_nodes:
            for child in node.flying_childrens:
                self.flying_edges.add((node.id, child.id))

    def reset_edges(self) -> None:
        self.edges = set()
        for node in self.nodes:
            for child in node.childrens:
                self.edges.add((node.id, child.id))
            for parent in node.parents:
                self.edges.add((parent.id, node.id))

    def get_flying_ids(self) -> List[str]:
        return [node.id for node in self.flying_nodes]

    def get_flying_modules(self) -> List[str]:
        return [node.name for node in self.flying_nodes]

    def delete_flying_node(self, node: Node) -> None:
        self.flying_nodes.remove(node)
        for flying_node in self.flying_nodes:
            if node in flying_node.flying_childrens:
                flying_node.flying_childrens.remove(node)
            if node in flying_node.flying_parents:
                flying_node.flying_parents.remove(node)

    def set_flying_relatives(self, node: Node) -> None:
        node.flying_childrens = []
        node.flying_parents = []
        if isinstance(node, ModuleNode):
            all_sub_childrens = node.all_sub_childrens
            all_sub_parents = node.all_sub_parents
        else:
            all_sub_childrens = node.next_ids
            all_sub_parents = node.previous_ids

        for flying_node in self.flying_nodes:
            if isinstance(flying_node, ModuleNode):
                all_root_sub_ids = flying_node.all_root_sub_ids
            else:
                all_root_sub_ids = [flying_node.id]

            if len(set(all_sub_childrens) & set(all_root_sub_ids)) > 0:
                node.flying_childrens.append(flying_node)
                if node not in flying_node.flying_parents:
                    flying_node.flying_parents.append(node)
            if len(set(all_sub_parents) & set(all_root_sub_ids)) > 0:
                node.flying_parents.append(flying_node)
                if node not in flying_node.flying_childrens:
                    flying_node.flying_childrens.append(node)

    def expend_flying_node(self, node_id:str) -> None:
        if node_id not in self.get_flying_ids():
            raise Exception("this node is not flying")
        node = self.get_node(node_id)
        if type(node) is ModuleNode:
            self.delete_flying_node(node)

            for new_node in node.sub_nodes:
                self.set_flying_relatives(new_node)
                self.flying_nodes.add(new_node)
            self.set_flying_edges()

        else:
            warnings.warn("this node is not can't be expended")

    def contract_flying_node(self, node_id:str) -> None:
        upper_node = self.get_node(node_id).upper_module
        if upper_node:
            sub_nodes = {
                flying_node
                for flying_node in self.flying_nodes
                if upper_node.id in flying_node.up_modules
            }

            for sub_node in sub_nodes:
                self.delete_flying_node(sub_node)

            self.set_flying_relatives(upper_node)
            self.flying_nodes.add(upper_node)
            self.set_flying_edges()

        else:
            warnings.warn("this node cannot be contract")

    def compute_flying_upper_modules_pos(self) -> None:
        self.flying_upper_modules = []

        upper_modules = {node.upper_module
                         for node in self.flying_nodes if node.upper_module is not None}

        for upper_module in upper_modules:
            upper_modules_pos = []
            for flying_node in self.flying_nodes:
                if upper_module.id in flying_node.up_modules:
                    upper_modules_pos.append(flying_node.pos)

            level = len(upper_module.up_modules)
            self.flying_upper_modules.append(
                create_bounding_rectangle(
                    rectangles=upper_modules_pos,
                    class_name=upper_module.module.__class__.__name__,
                    margin_width=20 * (self.level_max - level),
                    margin_height=20,
                    level=level
                )
            )

    def compute_pos_and_edges(self) -> None:
        G = AGraph(directed=True)

        for node in self.flying_nodes:
            if node.pos.width:
                G.add_node(node.id, shape="box", width=node.pos.width/72, height=node.pos.height/72)
            else:
                G.add_node(node.id, shape="box", width=1/72, height=1/72)

        for edge in self.flying_edges:
            G.add_edge(edge[0], edge[1], sep="0.001")

        G.layout(prog="dot")

        for node in G.nodes():
            graph_node = self.get_node(str(node), graph_name="flying")
            graph_node.pos.x = float(node.attr["pos"].split(",")[0])
            graph_node.pos.y = -float(node.attr["pos"].split(",")[1])

        self.pos_edges = []
        for edge in G.edges():
            edge_data = G.get_edge(edge[0], edge[1])
            pos = edge_data.attr["pos"]
            curve_points = parse_pos(pos)
            pose1 = self.get_node(edge[1])
            pose0 = self.get_node(edge[0])
            curve_points.insert(0, (pose0.pos.x, pose0.pos.y))
            curve_points.append((pose1.pos.x, pose1.pos.y))
            self.pos_edges.append(curve_points)

    def set_modules(self) -> None:
        for module_node in self.module_nodes:
            module_node.all_root_sub_ids = module_node.all_root_sub_ids & self.ids
            module_node.upper_module = None
            num_sub_nodes = float('inf')

            for node_include in self.module_nodes:
                if module_node is not node_include:
                    if module_node.all_root_sub_ids < node_include.all_root_sub_ids:
                        if (module_node.upper_module is None
                                or len(node_include.all_root_sub_ids) < num_sub_nodes):
                            module_node.upper_module = node_include
                            num_sub_nodes = len(node_include.all_root_sub_ids)

            if module_node.upper_module:
                module_node.upper_module.sub_nodes.add(module_node)

        for module_node in self.module_nodes:
            max_upper_module = module_node.upper_module
            while max_upper_module is not None:
                module_node.up_modules.append(max_upper_module.id)
                max_upper_module = max_upper_module.upper_module
            module_node.up_modules = module_node.up_modules[::-1]

    def set_nodes_in_module(self,
                            root_node_belong_to_module: Dict[str, str]) -> None:
        for node_id, module_id in root_node_belong_to_module.items():
            node = self.get_node(node_id)
            module = self.get_node(module_id, graph_name="module")
            if node:
                module.sub_nodes.add(node)
                node.up_modules = module.up_modules + [module_id]
                node.upper_module = module

        module_nodes_to_delete = {module_node for module_node in self.module_nodes
                                  if not module_node.sub_nodes}

        for module_node in module_nodes_to_delete:
            self.safe_delete(module_node)

    def set_number_parameters(self) -> None:
        for node in self.module_nodes:
            for idx in node.all_root_sub_ids:
                sub_node = self.get_node(idx)
                if isinstance(sub_node, VarNode):
                    node.nb_parameters += sub_node.variable.numel()

            self.max_number_parameters = max(self.max_number_parameters, node.nb_parameters)

    def set_height_and_width(self) -> None:
        for node in self.module_nodes:
            node.set_height_and_width(self.max_number_parameters)

    def set_wrapped_data(self,
                          wrapped_output:Dict) -> None:
        intermediate_nodes = []
        for wrapped_data in wrapped_output:
            node = self.get_node(wrapped_data["node"].id)
            if node:
                new_node = wrapped_data["node"]
                self.set_input(node, wrapped_data)

                new_node.up_modules = node.up_modules
                new_node.upper_module = node.upper_module
                self.safe_replace(node, new_node)

                parents_id = []
                for arg in wrapped_data["args"]+tuple(wrapped_data["kwargs"].values()):
                    if hasattr(arg, "grad_fn"):
                        if arg.grad_fn is not None:
                            parents_id.append(str(id(arg.grad_fn)))

                for parent_id in parents_id:
                    parent_node = self.get_node(parent_id)
                    if parent_node:
                        new_node.parents.append(parent_node)
                        parent_node.childrens.append(new_node)

                        for grad_fn_to_del in wrapped_data["grad_fn_created"]:
                            intermediate_nodes.append(self.get_node(grad_fn_to_del))
                            if grad_fn_to_del in parent_node.childrens:
                                parent_node.childrens.remove(grad_fn_to_del)

        for del_node in set(intermediate_nodes):
            if del_node in self.nodes:
                self.safe_delete(del_node)

    def safe_replace(self,
                     node:Node,
                     new_node:Node) -> None:

        new_node.childrens = node.childrens
        new_node.parents = node.parents

        for child in node.childrens:
            child.parents.append(new_node)

        for parent in node.parents:
            parent.childrens.append(new_node)

        if node.upper_module:
            node.upper_module.sub_nodes.remove(node)
            node.upper_module.sub_nodes.add(new_node)

        self.nodes.remove(node)
        self.nodes.add(new_node)

    def set_input(self, node: Node, wrap_data: Dict) -> None:
        args = wrap_data["args"]

        if all(isinstance(parent, VarNode) for parent in node.parents):
            params = [parent.variable for parent in node.parents]

            for arg in args:
                if isinstance(arg, Tensor) and not any(torch.equal(arg, t) for t in params):
                    input_node = VarNode(
                            id = str(id(arg)),
                            name = "input",
                            variable = arg,
                        )
                    input_node.add_children(node)
                    node.add_parent(input_node)
                    self.add_nodes(
                        input_node
                    )

    def set_up(self,
               hooks: Set[ModuleNode],
               root_node_belong_to_module: Dict[str, str],
               wrapped_output: Dict) -> None:

        self.add_nodes(hooks)
        self.module_nodes = hooks

        self.set_modules()
        self.set_nodes_in_module(root_node_belong_to_module)

        self.set_relatives()

        self.delete_nodes_type(ACCUMULATE_GRAD_FN)
        self.set_wrapped_data(wrapped_output)
        self.delete_nodes_type(NODE_FUNCTION_TO_DELETE)

        self.set_level_max()

        self.set_next_data()
        self.set_previous_data()

        self.set_sub_parents_and_childrens()

        self.set_number_parameters()
        self.set_height_and_width()

        self.init_flying_graph()
        self.set_flying_edges()

        self.compute_pos_and_edges()
        self.compute_flying_upper_modules_pos()
