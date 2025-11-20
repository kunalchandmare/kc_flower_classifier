import json

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

import helper as h


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch eval_model,
        returns an Numpy array
    '''
    # Load single image
    image = Image.open(image).convert('RGB')  # PIL Image

    # Same transforms as training (no augmentation for determinism)
    infer_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # Deterministic crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_xf = infer_transform(image).unsqueeze(0)
    return image_xf

def visualize_classifier(probs_list,top_class_id,image_show, classes_to_labels, true_label):

    top_classes = [classes_to_labels[idx] for idx in top_class_id]

    print(f"Top classes: {top_classes}")  # Debug
    print(f"Top probs: {probs_list}")
    # Create subplots: Left for image, right for bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,12))

    ax1 = h.imshow(image_show.squeeze(0),ax=ax1)
    ax1.set_title(f"Original Image\n(True Label: {true_label if true_label else 'Unknown'})")
    ax1.axis('off')

    # Right: Bar chart of top-k predictions
    y_pos = range(len(top_classes))
    colors = ['green' if cls == true_label else 'blue' for cls in top_classes]  # Highlight correct

    bars = ax2.barh(y_pos, probs_list, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Flowers')
    ax2.set_title('Model Predictions')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_classes, rotation=45, ha='right')

    # Add value labels on bars
    for bar, prob in zip(bars, probs_list):
        width = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2
        ax2.text(width +0.01,y_center,f'{prob:.3f}', ha='center',fontsize=10)

    plt.tight_layout(pad=2.0)
    plt.show()

def predict(image_path, inference_model, device, topk=5 ):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    inference_model.eval()
    proc_image = process_image(image_path)

    inference_model.to(device)
    proc_image.to(device)
    # Calculate the class probabilities (softmax) for img
    print("Processed but Unsqueezed Image Tensor:{}".format(proc_image.shape))
    with torch.no_grad():
        output = inference_model(proc_image)
        print(f"Output shape/range: {output.shape} / [{output.min():.2f}, {output.max():.2f}]")
        ps = torch.softmax(output, dim=1)  # Normalize to probabilities
        # if output.requires_grad or output.max() > 10:  # Heuristic: Raw logits can be large
        #     ps = torch.softmax(output, dim=1)  # FIXED: Normalize to probs [0,1]
        # else:
        #     ps = torch.exp(output)

        print(f"Probs sum: {ps.sum(dim=1)}")  # FIXED: Should be 1.0

        ps = ps.topk(topk, dim=1)
    return ps , proc_image
# Allowlist Sequential (do this once, before any load)
torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear,torch.nn.modules.activation.ReLU,
                                      torch.nn.modules.dropout.Dropout,
                                      torch.nn.modules.activation.LogSoftmax])
def main():
    in_arg = h.get_input_predict_args()
    image_path = in_arg.image_path
    checkpoint_path = in_arg.checkpoint
    top_k_num = in_arg.top_k
    cat_json_path = in_arg.category_names
    is_gpu = in_arg.gpu
    device = 'cuda' if torch.cuda.is_available() and is_gpu else 'cpu'

    # load model
    model, checkpoint= h.load_checkpoint(checkpoint_path)
    #print("Model loaded with Checkpoint: {}".format(checkpoint))
    model.eval()
    # Map indices to original class labels
    idx_to_class = {v: k for k, v in model.class_to_id.items()}

    image_label = h.get_label_from_filename(image_path)
    if cat_json_path and top_k_num:
        ps, proc_image = predict(image_path, model, device, top_k_num)
        top_prob, top_indices = ps.values, ps.indices
        top_prob_list = top_prob.squeeze().tolist()
        top_class_ids = [idx_to_class[i.item()] for i in top_indices.squeeze()]
        with open(cat_json_path, 'r') as f:
            class_to_name = json.load(f)
        #print(predictions)
        visualize_classifier(top_prob_list, top_class_ids, proc_image, class_to_name, true_label=image_path)
    else:
        ps, proc_image = predict(image_path, model, device)
        # Map indices to class labels
        top_prob, top_indices = ps.values, ps.indices
        top_prob_list = top_prob.squeeze().tolist()
        top_class_ids = [idx_to_class[i.item()] for i in top_indices.squeeze()]
        print("Flower {}: \n Class {} \n Probability {}".format(image_label, top_class_ids, top_prob_list))


if __name__ == "__main__":
    main()