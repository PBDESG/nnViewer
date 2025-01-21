# nnViewer

**nnViewer** is a Python library designed to provide an intuitive GUI for visualizing the structure and flow of a `torch.nn.Module`. Whether you're debugging or exploring complex neural networks, nnViewer makes it easier to understand your models.

## Installation

Install the library via pip:

```bash
pip install nnViewer
```

## Quick Start

Here's an example of how to use nnViewer with a Hugging Face model:

```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

from nnViewer.back.graph_initializer import wrap_model
from nnViewer.front.gui import run_gui

# Load an image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')

# Prepare the inputs
inputs = processor(images=image, return_tensors="pt")

# Initialize the graph
graph_init = wrap_model(model)

# Run the model to populate the graph
model(**inputs)

# Launch the GUI
run_gui(graph_init.graph)
```

## Overview

### `wrap_model(model: nn.Module) -> GraphInitializer`
Wraps a `torch.nn.Module` to initialize the computational graph for visualization.

### `run_gui(graph)`
Launches the GUI to display the computational graph.

## Contributing

Contributions are welcome! If you find any issues or have feature requests, feel free to open a GitHub issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
