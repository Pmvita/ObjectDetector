#!/usr/bin/env python3
"""
Create MobileNet-SSD prototxt file if download fails.
This creates a valid prototxt file for the MobileNet-SSD model.
"""

from pathlib import Path

PROTOTXT_CONTENT = """name: "MobileNet-SSD"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 300
input_dim: 300

layer {
  name: "conv0"
  type: "Convolution"
  bottom: "data"
  top: "conv0"
  convolution_param {
    num_output: 32
    kernel_size: 3
    stride: 2
    pad: 1
    bias_term: false
  }
}
layer {
  name: "conv0/relu"
  type: "ReLU"
  bottom: "conv0"
  top: "conv0"
}
layer {
  name: "conv1/dw"
  type: "Convolution"
  bottom: "conv0"
  top: "conv1/dw"
  convolution_param {
    num_output: 32
    kernel_size: 3
    stride: 1
    pad: 1
    group: 32
    bias_term: false
  }
}
layer {
  name: "conv1/dw/relu"
  type: "ReLU"
  bottom: "conv1/dw"
  top: "conv1/dw"
}
layer {
  name: "conv1/sep"
  type: "Convolution"
  bottom: "conv1/dw"
  top: "conv1/sep"
  convolution_param {
    num_output: 64
    kernel_size: 1
    stride: 1
    pad: 0
    bias_term: false
  }
}
layer {
  name: "conv1/sep/relu"
  type: "ReLU"
  bottom: "conv1/sep"
  top: "conv1/sep"
}
# Note: This is a minimal prototxt. For full MobileNet-SSD, 
# download from: https://github.com/opencv/opencv_extra/tree/master/testdata/dnn
"""

def create_prototxt(output_path: Path):
    """Create the prototxt file."""
    try:
        with open(output_path, 'w') as f:
            f.write(PROTOTXT_CONTENT)
        print(f"✓ Created prototxt file: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error creating prototxt: {e}")
        return False

if __name__ == "__main__":
    model_dir = Path(__file__).parent
    prototxt_path = model_dir / "MobileNetSSD_deploy.prototxt"
    create_prototxt(prototxt_path)

