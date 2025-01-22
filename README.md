
# nnViewer

**nnViewer** is a Python library designed to provide an intuitive GUI for visualizing the structure and flow of a `torch.nn.Module`. Whether you're debugging or exploring complex neural networks, nnViewer makes it easier to understand your models.

## Installation

Before installing `nnViewer`, you need to install `graphviz` and its development libraries. You can install them based on your operating system.

### Linux (Ubuntu/Debian)

1. **Install system dependencies**:
   ```bash
   sudo apt install graphviz libgraphviz-dev
   ```

2. **Install the Python package**:
   ```bash
   pip install nnViewer
   ```

### macOS

1. **Install system dependencies using Homebrew**:
   If you don't have Homebrew installed, you can install it from [here](https://brew.sh/). Then, install `graphviz`:
   ```bash
   brew install graphviz
   ```

2. **Install the Python package**:
   ```bash
   pip install nnViewer
   ```


## Quick Start

Here's an example of how to use nnViewer with a Hugging Face model:

```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
from nnViewer import wrap_model, run_gui

# Load an image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')

# Prepare the inputs
inputs = processor(images=image, return_tensors="pt")

# Wrap the model to initialize the graph
graph_init = wrap_model(model)

# Perform a forward pass to populate the graph
model(**inputs)

# Launch the GUI to visualize the computational graph
run_gui(graph_init.graph)
```

## Overview

### `wrap_model(model: nn.Module, fn_to_wrap: Optional[Callable] = None) -> GraphInitializer`

The `wrap_model` function wraps a `torch.nn.Module` and initializes the computational graph for visualization.

- **Arguments**:
  - `model` (`nn.Module`): The model to wrap.
  - `fn_to_wrap` (`Callable`, optional): A custom forward function to wrap. If not provided, the default `forward` function of the model is used. The output of the `fn_to_wrap` should be a tensor with a `grad_fn`, meaning the output must be a tensor that is part of the computation graph (i.e., a tensor that requires gradients).
  
- **Returns**: A `GraphInitializer` object that can be used to visualize the model's computational graph.

### How to Use

1. **Wrap the model**: First, you need to call `wrap_model(model)` to initialize the computational graph.
2. **Perform a forward pass**: Once the model is wrapped, run a forward pass with some sample inputs. This will populate the computational graph.
3. **Visualize the graph**: After the forward pass, you can visualize the model’s graph using the `run_gui()` function.

## Navigating the Graph

Once the GUI is running, you can interact with the computational graph in the following ways:

- **Click on a node**: Expands the node to show more information about the operation it represents.
- **Double-click on a node**: Contracts the node and hides the details, simplifying the view.
- **Right-click on a node**: Opens a context menu with options:
  - **Get More Information**: Shows additional details about the node, including its attributes. You can click on the attributes to view them.
  - **Show Computation**: Displays the matrices before and after the module for that specific node.

## Contributing

Contributions are welcome! If you find any issues or have feature requests, feel free to open a GitHub issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
