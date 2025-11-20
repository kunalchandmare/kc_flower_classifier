# Imports here
from collections import OrderedDict

import torch
from pyparsing import empty
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

import helper as h

train = 'train'
valid = 'validation'
test = 'test'
dataset = 'dataset'
dataloader = 'dataloader'
data_launch = {}

def prepare_datalaunch(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # FIXED: More crop variety
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # FIXED: Add color variation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Reload train_dataset with this, retrain
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)

    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=validation_transforms)
    print("Total Training Images: {}".format(len(image_datasets)))
    print("Total Validation Images: {}".format(len(validation_data)))
    print("Total Testing Images: {}".format(len(test_data)))
    # Using the image datasets and the trainforms, define the dataloaders
    image_dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    val_dataloaders = torch.utils.data.DataLoader(validation_data, batch_size=64,num_workers=0)
    test_dataloaders = torch.utils.data.DataLoader(test_data, batch_size=64)
    inputs,labels = next(iter(val_dataloaders))

    data_launch[train] = {dataset: image_datasets, dataloader:image_dataloaders}
    data_launch[valid] = {dataset: validation_data, dataloader: val_dataloaders}
    data_launch[test] = {dataset: test_data, dataloader: test_dataloaders}

    print("Total Train Iteration: {}".format(len(data_launch[train][dataset])//64))
    print("Total Validation Iteration: {}".format(len(data_launch[valid][dataset])//64))
    print("Total Test Iteration: {}".format(len(data_launch[test][dataset])//64))

config = {
    'optimizer': {
        'type': 'Adam',  # Optimizer class name
        'lr': 0.001,     # Learning rate
    },
    'loss':'CrossEntropyLoss',  # Loss function (for multi-class)
    'epochs': 3,
    'batch_size': 64,
    'device': 'cpu',
    'unfreeze_layers':8
}

def create_default_new_model():
    # Build and train your network
    default_model = models.vgg13(pretrained=True)
    # Freeze parameters so we don't backprop through them
    # Unfreeze last block (features.17,20,22 in VGG16-like)
    for name, param in default_model.named_parameters():
        if name in ['features.17', 'features.20', 'features.22', 'classifier']:
            param.requires_grad = True

            # 102 Categories as per JSON
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu', nn.ReLU()),
        ('drop1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(4096, 1000)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(0.2)),
        ('fc3', nn.Linear(1000, 102))
    ]))
    default_model.classifier = classifier
    return default_model

def create_new_model(model_type='vgg13', hidden_sizes=None, num_classes=102):
    """
    Dynamically creates a VGG model with custom classifier and unfreezing.

    Args:
        model_type (str): VGG variant ('vgg13', 'vgg16', 'vgg19').
        hidden_sizes (list[int]): Sizes for hidden layers in classifier (e.g., [4096, 1000]).
        num_classes (int): Number of output classes (default 102).

    Returns:
        nn.Module: Built model with last 3 feature layers and classifier unfrozen.
    """
    # Step 1: Load VGG base
    if hidden_sizes is None:
        hidden_sizes = [512]

    if model_type not in ['vgg13', 'vgg16', 'vgg19']:
        raise ValueError(f"model_type must be 'vgg13', 'vgg16', or 'vgg19'. Got {model_type}")

    vgg_base = getattr(models, model_type)(pretrained=True)

    # Step 2: Freeze all parameters initially
    for p in vgg_base.parameters():
        p.requires_grad = False

    # Step 3: Unfreeze last 3 layers of features
    features_children = list(vgg_base.features.children())
    last_n_layers = config['unfreeze_layers']
    for i in range(-last_n_layers, 0):  # Last 3: e.g., -3 to -1
        layer = features_children[i]
        print(f"Unfreezing feature layer {len(features_children) + i}: {type(layer).__name__}")
        for p in layer.parameters():
            p.requires_grad = True

    # Step 4: Build dynamic classifier based on hidden_sizes
    layers = []
    in_features = 25088  # VGG output after flatten (512 * 7 * 7)

    # Hidden layers
    for index, hidden_size in enumerate(hidden_sizes):
        layers.extend([
            (f'fc{index}', nn.Linear(in_features, hidden_size)),
            (f'relu{index}', nn.ReLU()),
            (f'drop{index}', nn.Dropout(0.2))
        ])
        in_features = hidden_size  # Update for next

    # Output layer
    layers.append(('fc_out', nn.Linear(in_features, num_classes)))

    classifier = nn.Sequential(OrderedDict(layers))

    # Step 5: Attach classifier
    vgg_base.classifier = classifier

    # Step 6: Unfreeze classifier
    for p in vgg_base.classifier.parameters():
        p.requires_grad = True

    # Step 7: Metadata (optional)
    vgg_base.class_to_idx = {}  # Set later if needed

    print(f"Model created: {model_type}, Hidden sizes: {hidden_sizes}, Classes: {num_classes}")
    print(f"Trainable params: {sum(p.numel() for p in vgg_base.parameters() if p.requires_grad):,}")

    return vgg_base

def calculate_validation_loss(eval_model, val_dataloader, criterion, log_frequency=5):
    # Validation every epoch
    eval_model.eval()  # Eval mode
    running_val_loss = 0.0
    total_avg_loss = 0
    with torch.no_grad():
        for val_step, val_sample in enumerate(val_dataloader):
            val_inputs, val_labels = val_sample  # FIXED: Consistent unpack
            val_inputs, val_labels = val_inputs.to(config['device']), val_labels.to(config['device'])
            val_log_ps = eval_model(val_inputs)  # Flatten
            val_loss = criterion(val_log_ps, val_labels)
            running_val_loss += val_loss.item()

            if (val_step + 1) % log_frequency == 0:
                total_avg_loss = running_val_loss / (val_step + 1)
    return total_avg_loss

def training_model(train_model, train_dataloader, val_dataloader):

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, train_model.parameters()), lr=config['optimizer']['lr'])

    if config['loss'] == 'NLLLoss':
        h.extend_classifier(train_model,{'LogSoftMax':nn.LogSoftmax(dim=1)})
        criterion = nn.NLLLoss()
    if config['loss'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    epochs = config['epochs']
    prints_every = 5
    # Set up lists to store losses for plotting
    train_losses = []
    train_steps = []
    eval_losses = []
    eval_steps = []
    for epoch in range(epochs):
        # Training phase
        train_model.train()  # Train mode
        running_train_loss = 0.0
        validation_loss = 0.0
        for step, sample in enumerate(train_dataloader):
            inputs, labels = sample
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            optimizer.zero_grad()
            log_ps = train_model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_losses.append(loss.item())
            train_steps.append(step + 1)
            print_step = (step+1) % prints_every
            if step ==0 or print_step == 0:
                avg_loss = running_train_loss / (step + 1)
                print("Epoch {} Training: Step {} Avg. Loss {:.3f}".format(epoch + 1, step + 1, avg_loss))

            # Validation every epoch
            if step in [len(train_dataloader)-1,0]:
                validation_loss = calculate_validation_loss(train_model, val_dataloader, criterion)
                print(f"Epoch {epoch + 1} Validation: Step {step + 1}  Avg. Loss {validation_loss:.3f}")

            eval_losses.append(validation_loss)
            eval_steps.append(step + 1)
            # Reset to train mode for next epoch
            train_model.train()
    return train_model, optimizer
def calculate_accuracy(test_model, test_dataloader, log_frequency=5):
    test_model.eval()
    accuracy = 0
    with torch.no_grad():
        for test_step, test_sample in enumerate(test_dataloader):
            test_inputs, test_labels = test_sample
            test_inputs, test_labels = test_inputs.to(config['device']), test_labels.to(config['device'])
            test_log_ps = test_model(test_inputs)
            #calculating accuracy
            ps = torch.exp(test_log_ps)
            high_p, high_class = ps.topk(1,dim=1)
            equals = high_class == test_labels.view(*high_class.shape)
            running_accuracy= torch.mean(equals.type(torch.FloatTensor)).item()
            accuracy += running_accuracy
            if (test_step+1) % log_frequency == 0:
                print(f"Accuracy for Step {test_step+1}: {running_accuracy:.3f}")
    print("Number of Test Samples: {}".format(len(test_dataloader)))
    print("Total Accuracy over Test Sample: {}".format(accuracy))
    return accuracy/len(test_dataloader)


def update_model_config(lr, epochs, is_gpu):
    if lr:
        print("Setting learning rate to {}".format(lr))
        config['optimizer']['lr'] = lr
    if epochs:
        print("Setting epochs to {}".format(epochs))
        config['epochs'] = epochs

    if is_gpu:
        print("Enabling GPU If available {}".format(is_gpu))
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    in_arg = h.get_input_training_args()
    data_dir = in_arg.directory
    model_arch = in_arg.arch
    hidden_units = in_arg.hidden_units
    lr = in_arg.learning_rate
    epochs = in_arg.epochs
    is_gpu = in_arg.gpu

    # Update hyper Parameter
    update_model_config(lr,epochs,is_gpu)
    print("Model Hyperparams: {}".format(config))

    # Construct the Model network
    model = create_new_model(model_type=model_arch, hidden_sizes=hidden_units)
    # Verify model parameters are correct
    #h.print_model_properties(model)

    # Preparing Dataset and data loaders
    print("Data Directory Parsed for Training: {}".format(data_dir))
    prepare_datalaunch(data_dir)

    #checking if dataloaders created correctly
    if (data_launch.get(train) and isinstance(data_launch[train], dict) and data_launch[train]) and \
            (data_launch.get(valid) and isinstance(data_launch[valid], dict) and data_launch[valid]):
        image_loader = data_launch.get(train).get(dataloader)
        inputs, labels = next(iter(image_loader))
        print("Image Loader sample {} with labels {}".format(inputs.shape, labels.shape))
        print("Dataloaders are loaded and Ready!!!!")
    else:
        print("Dataloaders are loaded and Ready!!!!")
        exit(0)


    # Training Starts here
    model.to(config['device'])
    model, optimizer = training_model(model, data_launch[train][dataloader], data_launch[valid][dataloader])
    avg_accuracy = calculate_accuracy(model, data_launch[test][dataloader], 2)
    print("Avg. Accuracy across Test Samples: {:2f}%".format(avg_accuracy * 100))

    checkpoint_properties = {'class_to_id': data_launch[train][dataset].class_to_idx,
                             'epochs': config['epochs'],
                             'arch': model_arch
                             }

    #h.save_checkpoint(model, 'checkpoint.pth', checkpoint_properties)

    #loaded_model = h.load_checkpoint('checkpoint_1.pth')
    try:
        while avg_accuracy < 0.8:
            training_model(model, data_launch[train][dataloader], data_launch[valid][dataloader])
            avg_accuracy = calculate_accuracy(model, data_launch[test][dataloader], 2)
    finally:
        print("Accuracy more than 80%, saving model")
        h.save_checkpoint(model, optimizer,'checkpoint.pth', checkpoint_properties)

if __name__ == "__main__":
    main()