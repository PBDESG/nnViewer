from PyQt5.QtWidgets import (QGraphicsRectItem, QMenu, QDialog, QLabel, QVBoxLayout, QFormLayout,
                             QPushButton,
                             QHBoxLayout, QTableWidget, QTableWidgetItem, QGraphicsDropShadowEffect,
                             QWidget, QFrame)
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter, QPainterPath
from torch import float16

from nnViewer.back.nodes import Conv2dNode, VarNode, ModuleNode, FonctionNode
from nnViewer.front.utils import (get_node_info, get_tensor_shape_as_string,
                                  get_image_from_slice_layout_and_tensor,
                                  set_up_slice_layout_from_tensor)


class ComputationCard(QFrame):
    def __init__(self, title, shape_text=None, parent=None):
        super().__init__(parent)
        self.setObjectName("computationCard")
        self.setStyleSheet("""
            QFrame#computationCard {
                background-color: #1a1a1a;
                border-radius: 15px;
                padding: 15px;
                min-width: 250px;
            }
            QLabel {
                color: #ffffff;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: none;
                border-radius: 5px;
                color: #ffffff;
                padding: 5px;
                min-width: 30px;
                max-width: 50px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #2196F3;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title_label.setStyleSheet("color: #2196F3;")
        layout.addWidget(title_label)

        # Shape text if provided
        if shape_text:
            shape_label = QLabel(shape_text)
            shape_label.setFont(QFont("Segoe UI", 10))
            shape_label.setStyleSheet("color: #808080;")
            layout.addWidget(shape_label)

        self.slice_container = QWidget()
        self.slice_layout = QFormLayout(self.slice_container)
        self.slice_layout.setSpacing(8)
        self.slice_layout.setContentsMargins(0, 5, 0, 5)
        layout.addWidget(self.slice_container)

        # Image container avec une taille plus grande
        self.image_scroll = QWidget()
        self.image_scroll.setFixedSize(250, 250)

        # Image container
        self.image_label = QLabel(self.image_scroll)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: transparent;")

        # Layout pour le widget d'image
        image_layout = QVBoxLayout(self.image_scroll)
        image_layout.addWidget(self.image_label)
        image_layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.image_scroll)

        # Effet d'ombre
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)

class SignalProxy(QObject):
    clicked = pyqtSignal(QGraphicsRectItem)
    doubleClicked = pyqtSignal(QGraphicsRectItem)


class ClickableRectItem(QGraphicsRectItem):
    def __init__(self, node, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.signal_proxy = SignalProxy()
        self.click_timer = QTimer()
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self.on_single_click)
        self.double_click_detected = False
        self.initial_pos = None
        self.as_moved = False
        self.node = node

    def mousePressEvent(self, event):
        self.initial_pos = self.pos()
        self.as_moved = False

    def mouseReleaseEvent(self, event):
        if self.click_timer.isActive():
            self.double_click_detected = True
            self.click_timer.stop()
        else:
            if event.button() == Qt.RightButton:
                self.show_context_menu(event.screenPos())
            else:
                self.double_click_detected = False
                self.click_timer.start(200)
                super().mousePressEvent(event)

        if self.pos() != self.initial_pos:
            self.as_moved = True

    def on_single_click(self):
        if not self.double_click_detected and not self.as_moved:
            self.signal_proxy.clicked.emit(self)

    def mouseDoubleClickEvent(self, event):
        self.signal_proxy.doubleClicked.emit(self)
        super().mouseDoubleClickEvent(event)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        pen = self.pen()
        brush = self.brush()
        painter.setPen(pen)
        painter.setBrush(brush)
        path = QPainterPath()
        rect = self.rect()
        path.addRoundedRect(rect, 15, 15)
        painter.drawPath(path)

    def show_context_menu(self, global_pos):
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a1a;
                border: 1px solid #2d2d2d;
                border-radius: 5px;
                color: #ffffff;
            }
            QMenu::item {
                padding: 8px 20px;
            }
            QMenu::item:selected {
                background-color: #2196F3;
            }
        """)

        expand_action = menu.addAction("Expand")
        contract_action = menu.addAction("Contract")
        info_action = menu.addAction("Get More Information")

        show_computation_action = None
        if isinstance(self.node, ModuleNode):
            show_computation_action = menu.addAction("Show Computation")

        if isinstance(self.node, VarNode):
            show_computation_action = menu.addAction("Show Variable")

        action = menu.exec(global_pos)

        if action == expand_action:
            self.signal_proxy.clicked.emit(self)
        elif action == contract_action:
            self.signal_proxy.doubleClicked.emit(self)
        elif action == info_action:
            self.get_more_information()
        elif action == show_computation_action:
            if not isinstance(self.node, FonctionNode):
                self.show_computation()

    def show_computation(self):
        if isinstance(self.node, Conv2dNode):
            self.show_conv2d_computation()
        elif isinstance(self.node, VarNode):
            self.show_var()
        else:
            self.show_default_computation()

    def create_computation_dialog(self, title):
        dialog = QDialog()
        dialog.setWindowFlag(Qt.FramelessWindowHint)

        # Style
        dialog.setStyleSheet("""
            QDialog {
                background-color: #121212;
                border-radius: 15px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

        # Header
        header = QWidget()
        header.setStyleSheet("background-color: #1a1a1a; border-radius: 15px 15px 0 0;")
        header_layout = QHBoxLayout(header)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #2196F3; font-size: 18px; font-weight: bold;")

        close_button = QPushButton("×")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #808080;
                font-size: 20px;
                font-weight: bold;
                padding: 5px 10px;
            }
            QPushButton:hover {
                color: #ff4444;
            }
        """)
        close_button.clicked.connect(dialog.close)

        header_layout.addWidget(title_label)
        header_layout.addWidget(close_button, alignment=Qt.AlignRight)

        # Main layout
        main_layout = QVBoxLayout(dialog)
        main_layout.addWidget(header)

        # Content container
        content = QWidget()
        main_layout.addWidget(content)

        # Shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 0)
        dialog.setGraphicsEffect(shadow)

        return dialog, content

    def show_conv2d_computation(self):
        dialog, content = self.create_computation_dialog("Convolution Layer Visualization")
        # dialog.setMinimumSize(1200, 900)

        layout = QHBoxLayout(content)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Input card
        input_card = ComputationCard("Input Tensor",
                                     f"Shape: {get_tensor_shape_as_string(self.node.input[0])}")
        input_slice_layout, _ = set_up_slice_layout_from_tensor(self.node.input[0])
        input_card.slice_layout.addRow(input_slice_layout)
        layout.addWidget(input_card)

        # Conv weights card
        conv_card = ComputationCard("Convolution Weights",
                                    f"Shape: {get_tensor_shape_as_string(self.node.module.weight)}")
        conv_slice_layout, _ = set_up_slice_layout_from_tensor(self.node.module.weight)
        conv_card.slice_layout.addRow(conv_slice_layout)
        layout.addWidget(conv_card)

        # Output card
        output_card = ComputationCard("Output Tensor",
                                      f"Shape: {get_tensor_shape_as_string(self.node.output[0])}")
        output_slice_layout, _ = set_up_slice_layout_from_tensor(self.node.output[0])
        output_card.slice_layout.addRow(output_slice_layout)
        layout.addWidget(output_card)

        # Update button with container
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        update_button = QPushButton("Update Visualization")
        button_layout.addWidget(update_button, alignment=Qt.AlignCenter)
        content.layout().addWidget(button_container)

        def update_displays():
            self.display_slices([
                (input_slice_layout, input_card.image_label, self.node.input[0], "gray"),
                (conv_slice_layout, conv_card.image_label, self.node.module.weight, "red"),
                (output_slice_layout, output_card.image_label, self.node.output[0], "gray")
            ])

        update_button.clicked.connect(update_displays)
        update_displays()  # Initial display

        dialog.exec_()

    def show_var(self):
        dialog, content = self.create_computation_dialog("Variable Visualization")

        layout = QVBoxLayout(content)

        # Variable card
        var_card = ComputationCard("Variable Tensor",
                                   f"Shape: {get_tensor_shape_as_string(self.node.variable)}")
        var_slice_layout, _ = set_up_slice_layout_from_tensor(self.node.variable)
        var_card.slice_layout.addRow(var_slice_layout)
        layout.addWidget(var_card)

        # Update button
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        update_button = QPushButton("Update Visualization")
        button_layout.addWidget(update_button)
        button_layout.setAlignment(Qt.AlignCenter)

        layout.addWidget(button_container)

        def update_displays():
            self.display_slices([
                (var_slice_layout, var_card.image_label, self.node.variable, "gray")
            ])

        update_button.clicked.connect(update_displays)
        update_displays()  # Initial display

        dialog.exec_()

    def show_default_computation(self):
        dialog, content = self.create_computation_dialog("Default Computation Visualization")

        layout = QHBoxLayout(content)
        layout.setSpacing(20)

        # Input card
        input_card = ComputationCard("Input Tensor",
                                     f"Shape: {get_tensor_shape_as_string(self.node.input[0])}")
        input_slice_layout, _ = set_up_slice_layout_from_tensor(self.node.input[0])
        input_card.slice_layout.addRow(input_slice_layout)
        layout.addWidget(input_card)

        # Output card
        output_card = ComputationCard("Output Tensor",
                                      f"Shape: {get_tensor_shape_as_string(self.node.output[0])}")
        output_slice_layout, _ = set_up_slice_layout_from_tensor(self.node.output[0])
        output_card.slice_layout.addRow(output_slice_layout)
        layout.addWidget(output_card)

        # Update button
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        update_button = QPushButton("Update Visualization")
        button_layout.addWidget(update_button)
        button_layout.setAlignment(Qt.AlignCenter)

        content.layout().addWidget(button_container)

        def update_displays():
            self.display_slices([
                (input_slice_layout, input_card.image_label, self.node.input[0], "gray"),
                (output_slice_layout, output_card.image_label, self.node.output[0], "gray")
            ])

        update_button.clicked.connect(update_displays)
        update_displays()  # Initial display

        dialog.exec_()

    def display_slices(self, matrix_to_display):
        try:
            for slice_layout, image_label, tensor, color in matrix_to_display:
                image = get_image_from_slice_layout_and_tensor(slice_layout, tensor.to(float16), color)

                parent_widget = image_label.parent()

                if parent_widget:
                    available_width = parent_widget.width() - 20
                    available_height = parent_widget.height() - 20

                    qimage = image.toImage()

                    scaled_image = qimage.scaled(available_width, available_height,
                                                 Qt.KeepAspectRatio,
                                                 Qt.SmoothTransformation)

                    image_label.setPixmap(QPixmap.fromImage(scaled_image))
                    image_label.setAlignment(Qt.AlignCenter)
                else:
                    image_label.setPixmap(image)
                    image_label.setAlignment(Qt.AlignCenter)

        except Exception as e:
            print(f"Error in displaying slices: {e}")

    def get_more_information(self):
        info_dialog = QDialog()
        info_dialog.setWindowTitle("Node Information")
        info_dialog.setWindowFlag(Qt.FramelessWindowHint)

        screen_geometry = info_dialog.screen().geometry()
        info_dialog.setGeometry(screen_geometry.width(), 50, 600, 400)

        table = QTableWidget(info_dialog)
        node_info = get_node_info(self.node)
        table.setRowCount(len(node_info))
        table.setColumnCount(2)

        table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                border: none;
                border-radius: 15px;
                color: #ffffff;
                gridline-color: #2d2d2d;
                selection-background-color: #2196F3;
            }
            QTableWidget::item {
                padding: 12px;
                border-bottom: 1px solid #2d2d2d;
            }
            QTableWidget::item:hover {
                background-color: #2d2d2d;
            }
            QScrollBar:vertical {
                border: none;
                background: #1a1a1a;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #404040;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4a4a4a;
            }
        """)

        if isinstance(self.node, ModuleNode):
            title_label = QLabel(f"Node Details - {self.node.module.__class__.__name__}")
        else:
            title_label = QLabel(f"Node Details - {self.node.name}")

        title_label.setStyleSheet("""
            QLabel {
                color: #2196F3;
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
                background-color: #1a1a1a;
                border-radius: 15px 15px 0 0;
            }
        """)

        close_button = QPushButton("×")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #808080;
                font-size: 20px;
                font-weight: bold;
                border: none;
                padding: 5px 10px;
            }
            QPushButton:hover {
                color: #ff4444;
            }
        """)
        close_button.clicked.connect(info_dialog.close)

        header_layout = QHBoxLayout()
        header_layout.addWidget(title_label)
        header_layout.addWidget(close_button, alignment=Qt.AlignRight)
        header_layout.setContentsMargins(0, 0, 0, 0)

        for row, (key, value) in enumerate(node_info.items()):
            key_item = QTableWidgetItem(f"  {key}")
            key_item.setBackground(QColor("#212121"))
            key_item.setForeground(QColor("#2196F3"))
            key_item.setFont(QFont("Segoe UI", 11, QFont.Bold))
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, key_item)

            if isinstance(value, dict) and "tensor" in value:
                value_item = CustomTableItem(value)
            else:
                value_item = QTableWidgetItem(str(value))

            value_item.setBackground(QColor("#1a1a1a"))
            value_item.setFont(QFont("Segoe UI", 11))
            value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 1, value_item)

        def on_cell_clicked(item):
            if isinstance(item, CustomTableItem) and item.tensor is not None:
                show_tensor_visualization(item.tensor, item.visualization_type, info_dialog)

        table.itemClicked.connect(on_cell_clicked)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)

        animation = QPropertyAnimation(info_dialog, b"geometry")
        animation.setDuration(300)
        animation.setStartValue(QRect(screen_geometry.width(), 50, 600, 400))
        animation.setEndValue(QRect(screen_geometry.width() - 620, 50, 600, 400))
        animation.setEasingCurve(QEasingCurve.OutCubic)

        main_layout = QVBoxLayout()
        main_layout.addLayout(header_layout)
        main_layout.addWidget(table)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(0)

        info_dialog.setLayout(main_layout)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 0)
        info_dialog.setGraphicsEffect(shadow)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        total_width = table.horizontalHeader().length() + 40
        total_height = table.verticalHeader().length() + 100
        info_dialog.setFixedSize(max(total_width, 400), max(total_height, 200))

        animation.start()
        info_dialog.exec_()


def show_tensor_visualization(tensor, visualization_type="tensor", parent=None):
    if tensor is None or not hasattr(tensor, 'shape'):
        return

    dialog = QDialog(parent)
    dialog.setWindowFlag(Qt.FramelessWindowHint)

    main_layout = QVBoxLayout(dialog)
    main_layout.setContentsMargins(10, 10, 10, 10)
    main_layout.setSpacing(0)

    header = QWidget()
    header.setStyleSheet("background-color: #1a1a1a; border-radius: 15px 15px 0 0;")
    header_layout = QHBoxLayout(header)

    title = "Kernel Visualization" if visualization_type == "kernel" else "Tensor Visualization"
    title_label = QLabel(title)
    title_label.setStyleSheet("color: #2196F3; font-size: 18px; font-weight: bold;")

    close_button = QPushButton("×")
    close_button.setStyleSheet("""
        QPushButton {
            background-color: transparent;
            color: #808080;
            font-size: 20px;
            font-weight: bold;
            padding: 5px 10px;
        }
        QPushButton:hover {
            color: #ff4444;
        }
    """)
    close_button.clicked.connect(dialog.close)

    header_layout.addWidget(title_label)
    header_layout.addWidget(close_button, alignment=Qt.AlignRight)

    main_layout.addWidget(header)

    # Content
    content = QWidget()
    content_layout = QVBoxLayout(content)

    # Shape information
    tensor_card = ComputationCard(title, f"Shape: ({', '.join(map(str, tensor.shape))})")
    tensor_slice_layout, _ = set_up_slice_layout_from_tensor(tensor)
    tensor_card.slice_layout.addRow(tensor_slice_layout)
    content_layout.addWidget(tensor_card)

    # Update button
    update_button = QPushButton("Update Visualization")
    content_layout.addWidget(update_button, alignment=Qt.AlignCenter)

    main_layout.addWidget(content)

    def update_display():
        color = "red" if visualization_type == "kernel" else "gray"
        try:
            image = get_image_from_slice_layout_and_tensor(tensor_slice_layout, tensor.to(float16), color)

            parent_widget = tensor_card.image_label.parent()
            if parent_widget:
                available_width = parent_widget.width() - 20
                available_height = parent_widget.height() - 20
                qimage = image.toImage()
                scaled_image = qimage.scaled(available_width, available_height,
                                             Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation)
                tensor_card.image_label.setPixmap(QPixmap.fromImage(scaled_image))
            else:
                tensor_card.image_label.setPixmap(image)

            tensor_card.image_label.setAlignment(Qt.AlignCenter)

        except Exception as e:
            print(f"Error in displaying tensor: {e}")

    update_button.clicked.connect(update_display)
    update_display()

    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(20)
    shadow.setColor(QColor(0, 0, 0, 150))
    shadow.setOffset(0, 0)
    dialog.setGraphicsEffect(shadow)

    dialog.exec_()

class CustomTableItem(QTableWidgetItem):
    def __init__(self, info_dict):
        super().__init__(info_dict.get("value", ""))
        self.tensor = info_dict.get("tensor")
        self.visualization_type = info_dict.get("visualization_type", "tensor")

        if self.tensor is not None:
            self.setToolTip("Click to visualize tensor")
            self.setForeground(QColor("#4FC3F7"))