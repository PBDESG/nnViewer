from typing import List, Tuple, Union

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QGraphicsRectItem
from torch import Tensor, nn, empty

from nnViewer.back.models import PosData

LEVEL_1_WIDTH = 250
MAX_HEIGHT = 500

class Node():
    def __init__(self,
                 id: str,
                 name: str):
        self.id = id
        self.name = name

        self.parents: List[Node] = []
        self.childrens: List[Node] = []
        self.flying_childrens: List[Node] = []
        self.flying_parents: List[Node] = []
        self.next_ids: List[str] = []
        self.previous_ids: List[str] = []
        self.pos: PosData = PosData()
        self.upper_module: ModuleNode = None
        self.up_modules = []
        self.item: QGraphicsRectItem = None
        self.color: QColor = QColor(0, 0, 0)

    def add_parent(self,
                    parent) -> None:
        self.parents.append(parent)

    def add_children(self,
                    children) -> None:
        self.childrens.append(children)

    def __hash__(self):
        return hash(self.id)


class VarNode(Node):
    def __init__(self,
                 id: str,
                 name: str,
                 variable: Tensor):

        super().__init__(id, name)
        self.variable = variable


class FonctionNode(Node):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 ):

        super().__init__(id, name)
        self.fonction = fonction

        self.output = None
        self.input = None
        self.parents_id = []

class ModuleNode(Node):
    def __init__(self,
                 id: str,
                 name: str,
                 input: Tuple[Tensor],
                 module: nn.Module,
                 output: Tuple[Tensor]):

        super().__init__(id, name)
        self.module = module
        self.input = input
        self.output = output

        self.sub_nodes = set()
        self.all_root_sub_ids = []
        self.all_sub_childrens = []
        self.all_sub_parents = []
        self.nb_parameters:int = 0

    def set_height_and_width(self, max_number_parameters):
        level = len(self.up_modules) - 2
        width = LEVEL_1_WIDTH * 0.8**level
        height = (self.nb_parameters/max_number_parameters) * MAX_HEIGHT
        self.pos.height = max(height, 25)
        self.pos.width = max(width, 25)

class LinearNode(ModuleNode):
    def __init__(self,
                 id: str,
                 name: str,
                 input: Tuple[Tensor],
                 module: nn.Module,
                 output: Tensor,
                 ):

        super().__init__(id, name, input, module, output)

        self.weight: Tensor = None
        self.bias: Tensor = None

    # def get_weights(self):
    #     for sub_node in self.sub_nodes:
    #         if sub_node.name == "weight":
    #             self.weight = sub_node.variable
    #         if sub_node.name== "bias":
    #             self.bias = sub_node.variable

    def set_height_and_width(self, max_number_parameters):
        self.pos.height = 70 * (1+(self.nb_parameters/max_number_parameters))
        self.pos.width = 50 * (1+(self.nb_parameters/max_number_parameters))

class LayerNormNode(ModuleNode):
    def __init__(self,
                 id: str,
                 name: str,
                 input: Tuple[Tensor],
                 module: nn.Module,
                 output: Tensor,
                 ):

        super().__init__(id, name, input, module, output)

        self.weight: Tensor = empty(())
        self.bias: Tensor = empty(())

    # def get_weights(self):
    #     for sub_node in self.sub_nodes:
    #         if sub_node.name == "weight":
    #             self.weight = sub_node.variable
    #         if sub_node.name == "bias":
    #             self.bias = sub_node.variable

class Conv2dNode(ModuleNode):
    def __init__(self,
                 id: str,
                 name: str,
                 input: Tuple[Tensor],
                 module: nn.Module,
                 output: Tuple[Tensor],
                 ):

        super().__init__(id, name, input, module, output)

class Conv1dNode(ModuleNode):
    def __init__(self,
                 id: str,
                 name: str,
                 input: Tuple[Tensor],
                 module: nn.Module,
                 output: Tuple[Tensor],
                 ):

        super().__init__(id, name, input, module, output)

class EmbeddingNode(ModuleNode):
    def __init__(self,
                 id: str,
                 name: str,
                 input: Tuple[Tensor],
                 module: nn.Module,
                 output: Tuple[Tensor],
                 ):

        super().__init__(id, name, input, module, output)

class BMMNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 mat1: Tensor,
                 mat2: Tensor,
                 output):

        super().__init__(id, name, fonction)
        self.mat2 = mat2
        self.mat1 = mat1
        self.output = output

class MulNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 mat1: Tensor,
                 mat2: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.mat1 = mat1
        self.mat2 = mat2
        self.output = output

class AttentionProductNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 key: Tensor,
                 query: Tensor,
                 value: Tensor,
                 output: Tensor,
                 mask: Union[Tensor, None]):

        super().__init__(id, name, fonction)
        self.key = key
        self.query = query
        self.value = value

        self.output = output
        self.mask = mask
        self.attention_matrix = query @ key.transpose(-2, -1)


class AddNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 mat1: Tensor,
                 mat2: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.mat1 = mat1
        self.mat2 = mat2
        self.output = output

class MatMulNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 mat1: Tensor,
                 mat2: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.mat1 = mat1
        self.mat2 = mat2
        self.output = output

class MeanNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 dim: int,
                 fonction: str,
                 input: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.input = input
        self.dim = dim
        self.output = output

class SumNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 dim: int,
                 fonction: str,
                 input: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.input = input
        self.dim = dim
        self.output = output

class PowNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 input: Tensor,
                 pow_value: float,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.input = input
        self.pow_value = pow_value
        self.output = output

class SubNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 mat1: Tensor,
                 mat2: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.mat1 = mat1
        self.mat2 = mat2
        self.output = output

class DivNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 mat1: Tensor,
                 mat2: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.mat1 = mat1
        self.mat2 = mat2
        self.output = output

class StackNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 input: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.input = input
        self.output = output

class ViewNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 input: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.input = input
        self.output = output

class TransposeNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 input: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.input = input
        self.output = output

class ExpandNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 input: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.input = input
        self.output = output

class ExpNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 exp_value: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.exp_value = exp_value
        self.output = output

class CatNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 input: Tensor,
                 output: Tensor):

        super().__init__(id, name, fonction)
        self.input = input
        self.output = output

class GetItemNode(FonctionNode):
    def __init__(self,
                 id: str,
                 name: str,
                 fonction: str,
                 input: Tensor,
                 output: Tensor,
                 slice: str):

        super().__init__(id, name, fonction)
        self.input = input
        self.output = output
        self.slice = slice
