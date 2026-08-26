import torch
from torch.utils.data import DataLoader, Subset
from captum.concept import TCAV, Concept
import os
import shutil
import torch




import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_loader_examples(pos_loader, neg_loader, n_images=32):
    """
    Display example images from the positive and negative concept loaders.

    Args:
        pos_loader: DataLoader for images with the concept.
        neg_loader: DataLoader for images without the concept.
        n_images: Number of images to display from each loader.
    """

    def get_images(loader):
        images = next(iter(loader))

        # Keep only the first n images
        images = images[:n_images]

        # Move to CPU for visualization
        return images.detach().cpu()

    pos_images = get_images(pos_loader)
    neg_images = get_images(neg_loader)

    # Create grids
    pos_grid = make_grid(pos_images, nrow=4, padding=2, normalize=True)
    neg_grid = make_grid(neg_images, nrow=4, padding=2, normalize=True)

    # Convert C,H,W -> H,W,C
    pos_grid = pos_grid.permute(1, 2, 0)
    neg_grid = neg_grid.permute(1, 2, 0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(pos_grid)
    axes[0].set_title("Positive / Has Concept")
    axes[0].axis("off")

    axes[1].imshow(neg_grid)
    axes[1].set_title("Negative / Does Not Have Concept")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()



from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from captum.concept import TCAV, Concept
import torch

def load_pure_concept_lib(concept_path, device):
    # Use the same transforms as your ZSL dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load images from the specific concept folder
    dataset = datasets.ImageFolder(concept_path, transform=transform)
    
    class StripLabelDataset(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i): return self.ds[i][0].to(device) # Strip label, keep on device

    return DataLoader(StripLabelDataset(dataset), batch_size=16, shuffle=True)


class Wrapper(torch.nn.Module):
    def __init__(self, lightning_model, target_prototype, target_class_idx, use_logits=False):
        super().__init__()
        self.cnn = lightning_model.cnn
        # The LTN predicate model
        self.ltn_classifier = lightning_model.embeddingFunction.isOfClass.model
        # The auxiliary linear classifier
        self.linear_classifier = getattr(lightning_model, 'classifier', None)
        
        self.target_prototype = target_prototype
        self.target_class_idx = target_class_idx
        self.use_logits = use_logits

    def forward(self, img):
        features = self.cnn(img)
        
        if self.use_logits:
            if self.linear_classifier is None:
                raise ValueError("Model does not have a linear classifier initialized.")
            # Predict logits and extract the single score for the target class
            logits = self.linear_classifier(features)
            return logits[:, self.target_class_idx].unsqueeze(1)
        else:
            # Calculate and return the LTN truth value
            return self.ltn_classifier(features, self.target_prototype).unsqueeze(1)


def run_tutorial_tcav_lib(lightning_model, data_module, target_class_name="zebra", use_logits=False):
    lightning_model.eval()
    device = lightning_model.device
    
    # 1. Load the Pure Concepts (from your custom folders)
    # Assumes you made: data/concepts/stripes/images/  and data/concepts/random/images/
    pos_loader = load_pure_concept_lib('./data/concepts/stripes', device)
    neg_loader = load_pure_concept_lib('./data/concepts/random', device)

    print(f"len(pos_loader): {len(pos_loader)}")
    print(f"len(neg_loader): {len(neg_loader)}")
    
    concept_target = Concept(id=0, name="stripes", data_iter=pos_loader)
    concept_random = Concept(id=1, name="random", data_iter=neg_loader)
    
    # 2. Setup the Wrapper
    class_idx = data_module.all_data['classes_names'].index(target_class_name)
    target_attributes = data_module.all_data['attributes_class_matrix'][class_idx].unsqueeze(0).to(device)
    
    with torch.no_grad():
        target_prototype = lightning_model.embeddingFunction(target_attributes)
        
    wrapper_model = Wrapper(
        lightning_model, 
        target_prototype, 
        target_class_idx=class_idx, 
        use_logits=use_logits
    ).to(device)
    
    # 3. Run Captum TCAV
    tcav = TCAV(model=wrapper_model, layers=['cnn.9'], save_path="./tcav_results_pure/") 
    
    # Target Zebras
    target_indices = [i for i, label in enumerate(data_module.ds_test_gzsl.labels) if label.item() == class_idx]
    target_subset = torch.utils.data.Subset(data_module.ds_test_gzsl, target_indices[:20])
    test_images = torch.stack([item[0] for item in target_subset]).to(device)
    test_images.requires_grad_() 
    
    scores = tcav.interpret(
        inputs=test_images, 
        experimental_sets=[[concept_target, concept_random]], 
        target=0 
    )
    
    # Print a helpful prefix so you know which path was evaluated
    eval_type = "Logits" if use_logits else "LTN Prototype"
    print(f"\nTCAV Scores ({eval_type}):")
    print(scores)




# -----------------------------------------------------------------------------

import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.linear_model import SGDClassifier

def get_concept_activations_hf(cnn, concept_path, device):
    """Loads pure concept images and extracts their final 2048-D embeddings."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(concept_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    embeddings = []
    with torch.no_grad():
        for imgs, _ in loader:
            # Output is already the final pooled (Batch, 2048) vector!
            emb = cnn(imgs.to(device))
            embeddings.append(emb.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)


def run_tcav_hf(lightning_model, data_module, target_class_name="zebra", attribute_name="stripes"):
    print(f"\n--- Running Custom TCAV: '{attribute_name.upper()}' on '{target_class_name.upper()}' ---")
    
    lightning_model.eval()
    device = lightning_model.device
    all_data = data_module.all_data

    # ==========================================
    # 1. Extract 2048-D Embeddings for Concepts
    # ==========================================
    print("Extracting 2048-D embeddings for pure concepts...")
    pos_path = f"./data/concepts/{attribute_name}"
    neg_path = "./data/concepts/random"
    
    pos_emb = get_concept_activations_hf(lightning_model.cnn, pos_path, device)
    neg_emb = get_concept_activations_hf(lightning_model.cnn, neg_path, device)

    # ==========================================
    # 2. Train Linear Classifier (The True CAV)
    # ==========================================
    print("Training Concept Vector (CAV)...")
    X = np.concatenate((pos_emb, neg_emb), axis=0)
    y = np.concatenate((np.ones(pos_emb.shape[0]), np.zeros(neg_emb.shape[0])))

    classifier = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3, random_state=42)
    classifier.fit(X, y)

    # The CAV is the orthogonal vector (Shape: [2048])
    CAV = torch.tensor(classifier.coef_[0], dtype=torch.float32, device=device)
    print(f"Concept Classifier Accuracy: {classifier.score(X, y) * 100:.1f}%")

    # ==========================================
    # 3. Get Gradients for the Target Class
    # ==========================================
    class_idx = all_data['classes_names'].index(target_class_name)
    target_indices = [i for i, label in enumerate(data_module.ds_test_gzsl.labels) if label.item() == class_idx]
    
    if len(target_indices) == 0:
        raise ValueError(f"No images found for '{target_class_name}'.")
        
    test_images = torch.stack([data_module.ds_test_gzsl[i][0] for i in target_indices[:50]]).to(device)
    target_attributes = all_data['attributes_class_matrix'][class_idx].unsqueeze(0).to(device)

    # Forward Pass through CNN (no gradients needed for the pixels)
    with torch.no_grad():
        visual_features = lightning_model.cnn(test_images) # Shape: [Batch, 2048]
    
    # ---> THE MAGIC STEP <---
    # We detach the 2048-D features and tell PyTorch to only track gradients from HERE forward
    visual_features = visual_features.detach()
    visual_features.requires_grad_()

    # Pass through LTN Logic
    prototype = lightning_model.embeddingFunction(target_attributes)
    truth_values = lightning_model.embeddingFunction.isOfClass.model(visual_features, prototype)

    # Backward Pass: Computes gradient of the Truth Value w.r.t the 2048-D visual features
    truth_values.sum().backward()
    
    # We extract the gradients directly from the tensor! (Shape: [Batch, 2048])
    gradients = visual_features.grad 

    # ==========================================
    # 4. Calculate Final TCAV Score
    # ==========================================
    # Dot product between [Batch, 2048] gradients and [2048] CAV vector
    directional_derivatives = torch.matmul(gradients, CAV)

    # The TCAV score is the % of test images where the derivative was positive
    tcav_score = (directional_derivatives > 0).float().mean().item()

    print(f"\nFinal TCAV Score: {tcav_score:.4f}")
    if tcav_score > 0.5:
        print(f"✅ The model actively relies on '{attribute_name}' to identify '{target_class_name}'.")
    else:
        print(f"❌ The model ignores '{attribute_name}' when identifying '{target_class_name}'.")