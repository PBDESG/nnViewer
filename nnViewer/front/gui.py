import sys

from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, QGraphicsTextItem
, QMainWindow, QGraphicsItem, QGraphicsPathItem)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QBrush, QPen, QPainter, QPainterPath, QFont, QColor
from PyQt5.QtCore import QTimer

from nnViewer.back.graph import Graph
from nnViewer.back.models import PosDataUpperModules
from nnViewer.back.nodes import (ModuleNode, FonctionNode, VarNode, LinearNode, Node,
                                 LayerNormNode, ViewNode, CatNode, AddNode, GetItemNode,
                                 Conv2dNode, MulNode, SubNode, Conv1dNode, DivNode, MatMulNode,
                                 AttentionProductNode)
from nnViewer.front.node_item import ClickableRectItem
from nnViewer.front.utils import create_centered_text_item, get_tensor_shape_as_string
from nnViewer.front.maps import map_strings_to_colors

FONT = "Courier New"


class GraphViewer(QMainWindow):
    def __init__(self, graph: Graph):
        super().__init__()

        self.setWindowTitle("Graph Viewer")
        screen_geometry = QApplication.desktop().screenGeometry()
        self.setGeometry(screen_geometry.x(), screen_geometry.y(), screen_geometry.width(), 800)

        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)

        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)

        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setInteractive(True)

        self.zoom_factor = 1.15

        self.graph = graph
        self.modules_colors = map_strings_to_colors(self.graph.get_module_class_name())
        self.rectangles = []

        self.set_items()
        self.render_graph()

    def render_graph(self):
        for edge in self.graph.pos_edges:
            self.draw_edge(edge)

        for node in self.graph.flying_nodes:
            self.draw_node(node)

        for upper_module in list(self.graph.flying_upper_modules):
            self.draw_upper_module(upper_module)

        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def draw_upper_module(self, upper_module: PosDataUpperModules):

        path = QPainterPath()
        rect = QRectF(
            upper_module.x - upper_module.width / 2,
            upper_module.y - upper_module.height / 2,
            upper_module.width,
            upper_module.height
        )
        corner_radius = 10
        path.addRoundedRect(rect, corner_radius, corner_radius)

        rect = QGraphicsPathItem(path)

        color = QColor(f"{self.modules_colors[upper_module.class_name]}")
        color.setAlpha(128)

        rect.setPen(QPen(Qt.black, 2))
        rect.setBrush(QBrush(color))
        rect.setZValue(upper_module.level)

        self.scene.addItem(rect)

        label = QGraphicsTextItem(upper_module.class_name)
        label.setFont(QFont(FONT, 12))

        nb_label = int(upper_module.height/(label.boundingRect().width()+1))
        label = QGraphicsTextItem(" ".join([upper_module.class_name]*nb_label))
        font = QFont(FONT, 12)
        font.setBold(True)
        label.setFont(font)

        label.setRotation(-90)

        label.setPos(
            upper_module.x - upper_module.width / 2,
            upper_module.y + label.boundingRect().width()/2
        )

        label.setZValue(upper_module.level)
        self.scene.addItem(label)

    def draw_node(self,
                  node: Node):

        rect = ClickableRectItem(node,
                                 node.pos.x - node.pos.width/2,
                                 node.pos.y - node.pos.height/2,
                                 node.pos.width,
                                 node.pos.height)
        rect.setFlag(QGraphicsItem.ItemIsSelectable)
        rect.setFlag(QGraphicsItem.ItemIsFocusable)
        rect.setPen(QPen(Qt.black))
        rect.setBrush(QBrush(node.color))
        rect.signal_proxy.clicked.connect(lambda: self.single_click_envent(rect))
        rect.signal_proxy.doubleClicked.connect(lambda: self.contract_graph(node))
        rect.setZValue(100)

        self.scene.addItem(rect)
        self.rectangles.append(rect)

        node.item.setPos(node.pos.x - node.item.boundingRect().width()/2,
                     node.pos.y - node.item.boundingRect().height()/2)
        node.item.setZValue(100)
        self.scene.addItem(node.item)

    def draw_edge(self, coordinates):
        path = QPainterPath()

        path.moveTo(coordinates[0][0], coordinates[0][1])
        for coordinate in coordinates[1:]:
            path.lineTo(coordinate[0], coordinate[1])

        pen = QPen(Qt.black, 2)
        path_item = self.scene.addPath(path, pen)
        path_item.setZValue(99)

    def set_items(self,
                  margin: int = 10):

        for node in self.graph.flying_nodes:
            node.color, node.item = self.get_illustration_item(node)

            # if isinstance(node, VarNode) or isinstance(node, FonctionNode):
            #     if node.name in function_mapping_plot.keys():
            #         margin = 0

            rect_width = node.item.boundingRect().width() + margin
            rect_height = node.item.boundingRect().height() + margin

            if isinstance(node, ModuleNode):
                rect_width = max(rect_width, node.pos.width)
                rect_height = max(rect_height, node.pos.height)

            node.pos.width = rect_width
            node.pos.height = rect_height

    def get_illustration_item(self, node):
        if isinstance(node, LinearNode):
            label_text = "Fully Connected Layer"
            color = QColor(250, 128, 114)
            item = QGraphicsTextItem(label_text)
            item.setFont(QFont(FONT, 12))
            return color, item

        elif isinstance(node, LayerNormNode):
            label_text = "Layer Normalization"
            color = QColor("#5b0e2d")
            item = QGraphicsTextItem(label_text)
            item.setFont(QFont(FONT, 12))
            return color, item

        elif isinstance(node, Conv2dNode):
            label_text = "Conv 2D"
            color = QColor("#7AAB9F")
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, Conv1dNode):
            label_text = "Conv 1D"
            color = QColor("#7AAB9F")
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, ModuleNode):
            label_text = node.module.__class__.__name__
            color = QColor(self.modules_colors[label_text])
            item = QGraphicsTextItem(label_text)
            level = len(node.up_modules) + 1
            item.setFont(QFont(FONT, int(12 + 12/level)))
            return color, item

        elif isinstance(node, ViewNode):
            label_text = (f"{get_tensor_shape_as_string(node.input)}\n"
                          f"->\n"
                          f"{get_tensor_shape_as_string(node.output)}")
            color = QColor("#26495c")
            item = create_centered_text_item(label_text, QFont(FONT, 12))

            return color, item

        elif isinstance(node, CatNode):
            str_shape = (get_tensor_shape_as_string(node.output))
            label_text = (f"concat \n"
                          f"{str_shape}")
            color = QColor("#26495c")
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, AddNode):
            label_text = "+"
            color = Qt.darkGray
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, DivNode):
            label_text = "/"
            color = Qt.darkGray
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, AttentionProductNode):
            label_text = "Attention Product"
            color = Qt.darkGray
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, MatMulNode):
            label_text = "Matrix Multiplication"
            color = Qt.darkGray
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, MulNode):
            label_text = "*"
            color = Qt.darkGray
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, SubNode):
            label_text = "-"
            color = Qt.darkGray
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, GetItemNode):
            label_text = node.slice
            color = Qt.darkGray
            item = create_centered_text_item(label_text, QFont(FONT, 12))
            return color, item

        elif isinstance(node, FonctionNode):
            label_text = node.name
            # if label_text in function_mapping_plot.keys():
            #     item = function_mapping_plot[label_text]()
            #     color = QColor(173, 216, 230)
            #     return color, item
            color = Qt.darkGray
            item = QGraphicsTextItem(label_text)
            item.setFont(QFont(FONT, 12))
            return color, item

        elif isinstance(node, VarNode):
            label_text = node.name
            color = QColor(93, 76, 61)
            item = QGraphicsTextItem(label_text)
            item.setFont(QFont(FONT, 12))
            return color, item

    def contract_graph(self, node):
        self.graph.contract_flying_node(node.id)
        self.set_items()
        self.graph.compute_pos_and_edges()
        self.graph.compute_flying_upper_modules_pos()
        QTimer.singleShot(0, self.clear_and_render_graph)

    def single_click_envent(self, rect):
        self.graph.expend_flying_node(rect.node.id)
        self.set_items()
        self.graph.compute_pos_and_edges()
        self.graph.compute_flying_upper_modules_pos()
        QTimer.singleShot(0, self.clear_and_render_graph)

    def clear_and_render_graph(self):
        self.clear_scene()
        self.render_graph()

    def clear_scene(self):
        for item in self.scene.items():
            if isinstance(item, ClickableRectItem):
                item.signal_proxy.clicked.disconnect()
        self.scene.clear()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.view.scale(self.zoom_factor, self.zoom_factor)
        elif event.key() == Qt.Key_Minus:
            self.view.scale(1 / self.zoom_factor, 1 / self.zoom_factor)

def run_gui(graph):
    app = QApplication(sys.argv)

    viewer = GraphViewer(graph)
    viewer.show()

    sys.exit(app.exec_())
