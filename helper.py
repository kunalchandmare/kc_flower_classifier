# Imports here
import argparse
import json
import os
from collections import OrderedDict

import matplotlib as plt
import numpy as np
import torch
from torch import nn
from torchvision import models


# Impo# Imports here

def check_model_on_gpu(model_to_check):
    return next(model_to_check.parameters()).is_cuda

def check_gpu_usage():
    # Check current memory
    if torch.cuda.is_available():
        print("=== GPU Memory Summary ===")
        print(torch.cuda.memory_summary(device=None, abbreviated=False))  # Detailed breakdown

        # Quick stats
        allocated = torch.cuda.memory_allocated() / 1024**3  # GiB
        reserved = torch.cuda.memory_reserved() / 1024**3     # GiB
        print(f"Allocated: {allocated:.2f} GiB")
        print(f"Reserved: {reserved:.2f} GiB")
    else:
        print("CUDA not available – on CPU")

def check_checkpoint_size(checkpoint_path):
    size_bytes = os.path.getsize(checkpoint_path)
    # Convert to human-readable
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    print("Model Size {} mb".format(size_mb))


def get_label_from_filename(filename):
    """
    Extracts label from file name (e.g., 'dog_456.jpg' → 'dog').

    Args:
        filename (str): Path or name of file.

    Returns:
        str: The label (prefix before first '_').
    """
    # Get base name (no path)
    base_name = os.path.basename(filename)

    # Split on first '_' (label is before it)
    label = base_name.split('_')[0]

    return label


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def load_checkpoint(filepath):
    print(os.getcwd())
    filepath = os.path.abspath(filepath)
    # Allowlist Sequential (do this once, before any load)
    torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear, torch.nn.modules.activation.ReLU,
                                          torch.nn.modules.dropout.Dropout,
                                          torch.nn.modules.activation.LogSoftmax,torch.nn.modules.container.Sequential])
    checkpoint = torch.load(filepath)
    arch = checkpoint.get('arch')
    # Dynamically create the correct VGG model default vgg13
    if arch is not None:
        if arch in ['vgg13', 'vgg16', 'vgg19']:
            loading_model = getattr(models, arch)(pretrained=True)
        else:
            raise ValueError(f"Unsupported architecture for loading: {arch}")
    else:
        loading_model = models.vgg13(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in loading_model.parameters():
        param.requires_grad = False

    loading_model.class_to_id = checkpoint.get('class_to_id', None)
    loading_model.classifier = checkpoint['classifier_layers']
    loading_model.load_state_dict(checkpoint['state_dict'])

    return loading_model, checkpoint


def save_checkpoint(trained_model,optimizer, relative_file_path, additional_properties):
    """
    :param trained_model:
    :param relative_file_path:
    :param additional_properties: Only epochs and class_to_id: Class id to reals cat na
    :return:
    """
    checkpoint = {'classifier_layers':trained_model.classifier,
                 'state_dict': trained_model.state_dict(),
                 'optimizer_state':optimizer.state_dict()}
    checkpoint.update(additional_properties)
    torch.save(checkpoint, relative_file_path)


def print_model_properties(model):
    """
    Prints all key properties of a PyTorch eval_model.

    Args:
        model (nn.Module): Your eval_model (e.g., VGGClassifier).
    """
    print("=== Model Overview ===")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n=== Parameters (Weights/Biases) ===")
    for name, param in model.named_parameters():
        print(f"  {name}: shape={param.shape}, trainable={param.requires_grad}, device={param.device}")

    print("\n=== Modules (Sub-Layers) ===")
    for name, module in model.named_modules():
        if name:  # Skip root
            print(f"  {name}: {module.__class__.__name__}")

    print("\n=== Buffers (Non-Trainable) ===")
    for name, buffer in model.named_buffers():
        print(f"  {name}: shape={buffer.shape}, device={buffer.device}")

    print("\n=== State Dict Keys (Checkpoint-Savable) ===")
    state_dict_keys = list(model.state_dict().keys())
    print(f"  Total keys: {len(state_dict_keys)}")
    print(f"  First 10: {state_dict_keys[:10]}")

def load_json(json_path):
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def get_input_training_args():
    """
    Retrieves and parses the 6 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 4 command line arguments where 1 argument MUST be Data Directory and
    Rest are optional parameters to save model and hyperparameters for model training. If the user fails to provide
    optional parameters, then the default values are used for the missing arguments.
    Command Line Arguments:
      1. Mandatory Image Folder as Data Directory to train the model on
      2. Optional Model Architecture as --arch with default value 'vgg13'
      3. Optional Model hyperparameter as --learning_rate with default value 0.01
      4. Optional Model hyperparameter as list --hidden_units with default value 512
      5. Optional Model hyperparameter as --epochs with default value 20
      6. Optional flag as --gpu with default value false
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(prog='Train on Images',
                                     description='Train on Images to classify Flower types with unfreezing last 8 layers'
                                                 'of the base model')
    parser.add_argument("directory", type=str, help='Path to the folder of flower images')
    parser.add_argument('--arch', type=str, default='vgg13', help='CNN Model architecture to use'
                                                                  'Note: only allowed vgg13, vgg16, vgg19')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', nargs="*", type=int, default=[4096,1000], help='Hidden units separated '
                                                                 'by space e.g. 1000 512 default [4096,1000]')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs default 5')
    parser.add_argument('--gpu', action="store_true", help='Use GPU for training')
    return parser.parse_args()


def get_input_predict_args():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 4 command line arguments where 1 argument MUST be Image Directory to
    predict and  Rest are optional parameters to see more details e.g. Top 3 classes predicted and provide json from
    which grab class id to real cat names optional parameters, then the default values are used for the missing arguments.
    Command Line Arguments:
      1. Mandatory Image as --img path to the flower image
      2. Mandatory path to checkpoint of Model to load
      3. Optional --top_k define Number of top Class to show default 3
      4. Optional --category_names JSON file path with mapping to actual cat names
      5. Optional flag as --gpu with default value false
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(prog='Predict Image',
                                    description='Predict image to classify Flower types')
    parser.add_argument("image_path", type=str, help='Path to image to predict must be label.jpg '
                                                     'e.g. mexican aster.jpg')
    parser.add_argument("checkpoint", type=str, help='Path to model checkpoint file .pth'
                                                     'Note: only supports vgg13, vgg16, vgg19')
    parser.add_argument('--top_k', type=int, help='Number of top classes to show default 5')
    parser.add_argument('--category_names', type=str,
                        help='JSON file for category names')
    parser.add_argument('--gpu', action="store_true", help='Use GPU for inference')
    return parser.parse_args()


def extend_classifier(model, new_layer: dict):
    """
    Extends model.classifier with a new layer.

    Args:
        model (nn.Module): Your VGG model.
        new_layer (nn.Module): The layer to add (e.g., nn.Linear(102, 50)).

    Returns:
        nn.Module: Model with extended classifier.
    """
    # Step 1: Extract original layers as OrderedDict
    original_layers = OrderedDict(model.classifier.named_children())

    for key, value in new_layer.items():
        # Step 2: Add new layer (with name, e.g., 'new_fc')
        original_layers[key] = value

    # Step 3: Rebuild Sequential
    extended_classifier = nn.Sequential(original_layers)

    # Step 4: Attach to model
    model.classifier = extended_classifier

    print(f"Extended classifier with {new_layer}")
    print(model.classifier)
    return model


def check_gradient_change(step, model, loss, initial_weight):
    # After loss.backward()
    if step % 10 == 0:  # Every 10 steps to avoid spam
        print(f"Step {step}: Loss {loss.item():.3f}")

        # FIXED: Check grad norm on a param (e.g., classifier.fc1.weight)
        if model.classifier.fc1.weight.grad is not None:
            grad_norm = torch.norm(model.classifier.fc1.weight.grad).item()
            print(f"Grad norm (fc1): {grad_norm:.3f}")  # Expected: 0.1-10 (healthy)
        else:
            print("WARNING: No gradients on fc1 – check requires_grad")

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"{name}: grad norm={param.grad.norm().item():.4f}")

    # Before/after 1 epoch check (run once)
    if step == 0:
        initial_weight = model.classifier.fc1.weight.clone().detach()

    if step == 9:  # After 10 steps
        weight_change = torch.norm(model.classifier.fc1.weight - initial_weight).item()
        print(f"Weight change: {weight_change:.3f}")

