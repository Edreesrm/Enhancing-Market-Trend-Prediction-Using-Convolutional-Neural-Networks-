import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from typing import Dict, List, Tuple
import math

def get_model_memory_size(model: Model) -> float:
    """Calculate the total size of the model in MB."""
    import tempfile
    import os
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Save model with .keras extension
        model_path = os.path.join(temp_dir, 'model.keras')
        model.save(model_path)
        size = os.path.getsize(model_path)
        size_mb = size / (1024 * 1024)
    finally:
        shutil.rmtree(temp_dir)
    
    return size_mb

def analyze_cnn_complexity(model: Model, input_shape: Tuple[int, ...]) -> Dict:
    """
    Analyzes the complexity of a CNN model using TensorFlow/Keras.
    
    Args:
        model: Keras model
        input_shape: Input shape (height, width, channels) or (batch_size, height, width, channels)
        
    Returns:
        Dictionary containing various complexity metrics
    """
    # Ensure input_shape has batch dimension
    if len(input_shape) == 3:
        input_shape = (1,) + input_shape
        
    # Build the model with the input shape
    if not model.built:
        model.build(input_shape)
        
    def get_flops_conv2d(layer: tf.keras.layers.Conv2D, input_shape: Tuple) -> int:
        """Calculate FLOPs for Conv2D layer."""
        kernel_size = layer.kernel_size
        filters = layer.filters
        input_channels = input_shape[-1]
        
        # Calculate output dimensions
        if layer.padding == 'same':
            output_height = math.ceil(input_shape[1] / layer.strides[0])
            output_width = math.ceil(input_shape[2] / layer.strides[1])
        else:  # 'valid' padding
            output_height = math.ceil((input_shape[1] - kernel_size[0] + 1) / layer.strides[0])
            output_width = math.ceil((input_shape[2] - kernel_size[1] + 1) / layer.strides[1])
            
        # FLOPs for convolution = H * W * Cin * Cout * Kh * Kw
        flops = (output_height * output_width * 
                input_channels * filters * 
                kernel_size[0] * kernel_size[1])
        
        # Add FLOPs for bias if used
        if layer.use_bias:
            flops += output_height * output_width * filters
            
        return flops, (input_shape[0], output_height, output_width, filters)
    
    def get_flops_dense(layer: tf.keras.layers.Dense, input_shape: Tuple) -> int:
        """Calculate FLOPs for Dense layer."""
        input_features = np.prod(input_shape[1:])  # Flatten input if needed
        output_features = layer.units
        
        # FLOPs for dense layer = input_features * output_features
        flops = input_features * output_features
        if layer.use_bias:
            flops += output_features
            
        return flops, (input_shape[0], output_features)
    
    def get_layer_info(layer: tf.keras.layers.Layer, input_shape: Tuple) -> Dict:
        """Calculate parameters and operations for a single layer."""
        layer_type = layer.__class__.__name__
        params = layer.count_params()
        flops = 0
        output_shape = input_shape
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            flops, output_shape = get_flops_conv2d(layer, input_shape)
            
        elif isinstance(layer, tf.keras.layers.Dense):
            flops, output_shape = get_flops_dense(layer, input_shape)
            
        elif isinstance(layer, (tf.keras.layers.ReLU, tf.keras.layers.Activation)):
            flops = np.prod(input_shape)  # One operation per element
            output_shape = input_shape
            
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            output_height = math.ceil(input_shape[1] / layer.pool_size[0])
            output_width = math.ceil(input_shape[2] / layer.pool_size[1])
            output_shape = (input_shape[0], output_height, output_width, input_shape[3])
            flops = (np.prod(layer.pool_size) * 
                    output_height * output_width * input_shape[3])
            
        elif isinstance(layer, tf.keras.layers.Flatten):
            output_shape = (input_shape[0], np.prod(input_shape[1:]))
            flops = 0  # Flatten is just a reshape operation
            
        return {
            'type': layer_type,
            'params': params,
            'flops': flops,
            'input_shape': input_shape,
            'output_shape': output_shape
        }

    # Initialize metrics
    total_params = 0
    total_flops = 0
    layer_metrics = []
    current_shape = input_shape
    
    # Analyze each layer
    for layer in model.layers:
        metrics = get_layer_info(layer, current_shape)
        total_params += metrics['params']
        total_flops += metrics['flops']
        layer_metrics.append(metrics)
        current_shape = metrics['output_shape']
    
    # Calculate memory footprint (in MB)
    param_memory = total_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
    
    # Calculate model depth (excluding activation and pooling layers)
    depth = len([l for l in model.layers 
                 if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))])
    
    try:
        # Get total size of the model
        model_size = get_model_memory_size(model)
    except Exception as e:
        print(f"Warning: Could not calculate model size: {str(e)}")
        model_size = 0
    
    return {
        'total_parameters': total_params,
        'total_flops': total_flops,
        'parameter_memory_mb': param_memory,
        'model_size_mb': model_size,
        'model_depth': depth,
        'layer_metrics': layer_metrics
    }

def print_complexity_analysis(analysis: Dict):
    """Pretty print the complexity analysis results."""
    print("\nCNN Model Complexity Analysis")
    print("=" * 50)
    print(f"Total Parameters: {analysis['total_parameters']:,}")
    print(f"Total FLOPs: {analysis['total_flops']:,}")
    print(f"Parameter Memory: {analysis['parameter_memory_mb']:.2f} MB")
    if analysis['model_size_mb'] > 0:
        print(f"Model Size: {analysis['model_size_mb']:.2f} MB")
    print(f"Model Depth: {analysis['model_depth']} layers")
    
    print("\nLayer-wise Analysis:")
    print("-" * 80)
    print(f"{'Layer Type':<15} {'Parameters':<12} {'FLOPs':<15} {'Output Shape':<20}")
    print("-" * 80)
    
    for layer in analysis['layer_metrics']:
        print(f"{layer['type']:<15} {layer['params']:<12,} {layer['flops']:<15,} {str(layer['output_shape']):<20}")

def measure_inference_time(model: Model, 
                         input_shape: Tuple[int, ...],
                         num_iterations: int = 100) -> Dict:
    """
    Measure model inference time over multiple iterations.
    
    Args:
        model: Keras model
        input_shape: Input shape including batch dimension
        num_iterations: Number of inference iterations for averaging
        
    Returns:
        Dictionary containing timing metrics
    """
    # Create random input data
    input_data = np.random.random(input_shape)
    
    # Warmup run
    model.predict(input_data, verbose=0)
    
    # Measure inference time
    times = []
    for _ in range(num_iterations):
        start_time = tf.timestamp()
        model.predict(input_data, verbose=0)
        end_time = tf.timestamp()
        times.append(float(end_time - start_time))
    
    return {
        'mean_inference_time': np.mean(times),
        'std_inference_time': np.std(times),
        'min_inference_time': np.min(times),
        'max_inference_time': np.max(times)
    }

def print_inference_analysis(timing: Dict):
    """Pretty print the inference timing results."""
    print("\nInference Time Analysis")
    print("=" * 50)
    print(f"Mean Inference Time: {timing['mean_inference_time']*1000:.2f} ms")
    print(f"Std Inference Time: {timing['std_inference_time']*1000:.2f} ms")
    print(f"Min Inference Time: {timing['min_inference_time']*1000:.2f} ms")
    print(f"Max Inference Time: {timing['max_inference_time']*1000:.2f} ms")
    
    
# Example usage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout
from tensorflow.keras.layers import Input

# Create a sample model


model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'), 
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
# Analyze model complexity
input_shape = (64, 150, 150, 3)  # Include batch dimension
analysis = analyze_cnn_complexity(model, input_shape)
print_complexity_analysis(analysis)

# Measure inference time
timing = measure_inference_time(model, input_shape)
print_inference_analysis(timing)