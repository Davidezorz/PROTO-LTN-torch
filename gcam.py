import torch
import numpy as np
import ltn

import cv2
import matplotlib.pyplot as plt

class LTN_GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture forward and backward pass data
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        # Hook to capture the gradients after the backward pass
        self.gradients = grad_output[0].detach()

    def forward_hook(self, module, input, output):
        # Hook to capture the activations after the forward pass
        self.activations = output.detach()

    def generate_cam(self, input_image, target_class_idx, use_logits=False):
        self.model.zero_grad()
        
        # 1. Forward pass through the CNN to extract visual features
        features = self.model.cnn(input_image)
        
        if use_logits:
            # Ensure the model actually has a classifier initialized
            if not hasattr(self.model, 'classifier') or self.model.classifier is None:
                raise ValueError("Cannot compute CAM w.r.t logits: classifier is not initialized.")
            
            # Predict logits directly from the CNN features
            logits = self.model.classifier(features)
            
            # Select the logit corresponding to the target class
            target_score = logits[0, target_class_idx]
            
        else:
            # 2. Extract the prototype for the target class (e.g., Zebra)
            target_attributes = self.model.attributes_class_matrix[target_class_idx].unsqueeze(0)
            prototype = self.model.embeddingFunction(target_attributes)
            
            # 3. Calculate the LTN truth value (our "class score")
            feature_var = ltn.Variable("feature", features)
            prototype_var = ltn.Variable("prototype", prototype)
            
            # The truth value is a measure of similarity [0, 1] between the image and the prototype
            target_score = self.model.embeddingFunction.isOfClass(feature_var, prototype_var).value
        
        # 4. Backpropagate to get gradients w.r.t. the chosen target score
        target_score.backward(retain_graph=True)
        
        # 5. Generate CAM by weighting the activations by the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Clone to avoid in-place modification of hook outputs
        weighted_activations = self.activations.clone() 
        for i in range(weighted_activations.shape[1]):
            weighted_activations[:, i, :, :] *= pooled_gradients[i]

        cam = torch.mean(weighted_activations, dim=1).squeeze().cpu().numpy()
        print(f"cam < 0: {(cam[cam<0]).sum()}")
        cam = np.maximum(cam, 0)  # Apply ReLU
        
        # Normalize the heatmap to [0, 1] securely
        if cam.max() - cam.min() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
        return cam




def plot_gcam(model, data_module, device, use_logits=False, random_idx=None):
    # 1. Find the index of the "zebra" class in your data
    zebra_idx = data_module.all_data['classes_names'].index('zebra')
    model.cnn.requires_grad_(True)

    # 2. Select the target layer
    # In your ResNet101 Sequential setup, index 7 is the final `layer4` conv block.
    target_layer = model.cnn[7]

    # 3. Initialize GradCAM
    grad_cam = LTN_GradCAM(model, target_layer)

    # =====================================================================
    # 3.5 Fetch a random Zebra image from the training set
    # =====================================================================
    # Find all indices in the training dataset where the label matches zebra_idx
    zebra_indices = (data_module.ds_train.labels == zebra_idx).nonzero(as_tuple=True)[0]

    # Pick a random index from the available zebra images 0, 7 8, 9
    if random_idx is None:
        random_idx = zebra_indices[torch.randint(len(zebra_indices), (1,))].item()
    else:
        random_idx = zebra_indices[random_idx]

    # Extract the image tensor (shape: [3, 224, 224]), label, and attributes[cite: 3]
    input_image_tensor, label, _ = data_module.ds_train[random_idx]

    # Add the batch dimension (shape: [1, 3, 224, 224]) and move to device
    input_image = input_image_tensor.unsqueeze(0).to(device)
    input_image.requires_grad_(True)

    # 4. Generate the heatmap
    cam_heatmap = grad_cam.generate_cam(input_image, target_class_idx=zebra_idx,
                                        use_logits=use_logits)

    print(f"cam_heatmap all 0s: {np.allclose(cam_heatmap, np.zeros_like(cam_heatmap))}")
    print(f"cam_heatmap sum:    {(np.abs(cam_heatmap)).sum(): .16f}")
    # =====================================================================
    # 5. Visualize (Overlaying on the original image)
    # =====================================================================
    # Resize the heatmap to match the image dimensions
    cam_heatmap_resized = cv2.resize(cam_heatmap, (224, 224))

    # Denormalize the image tensor for displaying[cite: 3]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert tensor to numpy and rearrange dimensions from [C, H, W] to [H, W, C]
    img_display = input_image_tensor.permute(1, 2, 0).cpu().numpy()
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1) # Ensure values are strictly between 0 and 1

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(img_display)
    plt.imshow(cam_heatmap_resized, cmap='jet', alpha=0.5) # Overlay the heatmap
    plt.axis('off')
    plt.title(f"Grad-CAM: Zebra (Index: {random_idx})")
    plt.show()