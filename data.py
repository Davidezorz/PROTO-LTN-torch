import os
import urllib.request
import zipfile
import scipy.io
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
import torchvision.transforms as transforms


# ╭───────────────────────────────────────────────────────────────────────────╮
# │                              PyTorch Datasets                             │
# ╰───────────────────────────────────────────────────────────────────────────╯

class ZSLDataset(Dataset):

    def __init__(self, features, labels, attributes):
        self.features   = torch.tensor(features, dtype=torch.float32)
        self.labels     = torch.tensor(labels, dtype=torch.long)
        self.attributes = torch.tensor(attributes, dtype=torch.float32)


    def __len__(self):
        return self.features.shape[0]


    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.attributes[idx]





class ZSLImageDataset(Dataset):

    def __init__(self, image_paths, labels, attributes, is_train=True):
        self.image_paths = image_paths
        self.labels      = torch.tensor(labels, dtype=torch.long)
        self.attributes  = torch.tensor(attributes, dtype=torch.float32)
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx], self.attributes[idx]






# ╭───────────────────────────────────────────────────────────────────────────╮
# │                       PyTorch Lightning DataModule                        │
# ╰───────────────────────────────────────────────────────────────────────────╯

class ZSLDataModule(pl.LightningDataModule):

    def __init__(self, base_path='./data', dataset_name='AWA2', batch_size=32, num_workers=4, use_raw_images=False):
        super().__init__()
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_raw_images = use_raw_images 

        self.url = "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
        self.dataset_dir = os.path.join(self.base_path, 'xlsa17', 'data', self.dataset_name)


    def prepare_data(self):
        if not os.path.exists(self.dataset_dir):
            print(f"Downloading dataset {self.dataset_name} to {self.base_path}...")
            os.makedirs(self.base_path, exist_ok=True)
            zip_path = os.path.join(self.base_path, 'xlsa17.zip')
            
            if not os.path.exists(zip_path):
                urllib.request.urlretrieve(self.url, zip_path)
                
            print("Extracting... (this might take a minute)")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_path)
            print("Dataset ready!")


    def setup(self, stage=None):
        res101 = scipy.io.loadmat(os.path.join(self.dataset_dir, 'res101.mat'))
        att_splits = scipy.io.loadmat(os.path.join(self.dataset_dir, 'att_splits.mat'))

        all_labels = (res101['labels'] - 1).astype(int).squeeze() 
        classes_names = [att_splits['allclasses_names'][i][0][0] for i in range(att_splits['allclasses_names'].size)]
        
        attributes_class_matrix = np.transpose(att_splits['att'])
        
        test_unseen = att_splits['test_unseen_loc'].squeeze() - 1
        test_seen = att_splits['test_seen_loc'].squeeze() - 1
        test = np.concatenate((test_unseen, test_seen))
        train = att_splits['trainval_loc'].squeeze() - 1
        
        attribute = att_splits['original_att'].T  

        # --- DYNAMIC ATTRIBUTES LOADING ---
        predicates_path = os.path.join(self.base_path, 'Animals_with_Attributes2', 'predicates.txt')
        attribute_names = []
        if os.path.exists(predicates_path):
            with open(predicates_path, 'r') as f:
                for line in f:
                    # Each line looks like "1 black". We split by whitespace and take the second item.
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        attribute_names.append(parts[1])
        else:
            print(f"Warning: {predicates_path} not found. Attribute names will be empty.")

        # --- BRANCH: IMAGES VS FEATURES ---
        if self.use_raw_images:
            raw_paths = res101['image_files']
            local_base = os.path.join(self.base_path, 'Animals_with_Attributes2') 
            cleaned_paths = []
            
            for path in raw_paths.squeeze():
                raw_path_str = str(path[0])
                # Safely split at 'JPEGImages' to remove the author's Linux absolute path
                relative_path = 'JPEGImages' + raw_path_str.split('JPEGImages')[-1]
                # To handle that weird double slash "//" the author accidentally included:
                relative_path = relative_path.replace('//', '/')
                
                cleaned_paths.append(os.path.join(local_base, relative_path))
                
            cleaned_paths = np.array(cleaned_paths)

            if stage == 'fit' or stage is None:
                self.ds_train = ZSLImageDataset(cleaned_paths[train], all_labels[train], attributes_class_matrix[all_labels[train]], is_train=True)
            if stage in ['test', 'validate', 'val', None]:
                self.ds_test_zsl = ZSLImageDataset(cleaned_paths[test_unseen], all_labels[test_unseen], attributes_class_matrix[all_labels[test_unseen]], is_train=False)
                self.ds_test_gzsl = ZSLImageDataset(cleaned_paths[test], all_labels[test], attributes_class_matrix[all_labels[test]], is_train=False)
        else:
            all_features = np.transpose(res101['features'])
            if stage == 'fit' or stage is None:
                self.ds_train = ZSLDataset(all_features[train], all_labels[train], attributes_class_matrix[all_labels[train]])
            if stage in ['test', 'validate', 'val', None]:
                self.ds_test_zsl = ZSLDataset(all_features[test_unseen], all_labels[test_unseen], attributes_class_matrix[all_labels[test_unseen]])
                self.ds_test_gzsl = ZSLDataset(all_features[test], all_labels[test], attributes_class_matrix[all_labels[test]])

        # Global properties
        test_unseen_classes = torch.tensor(np.unique(all_labels[test_unseen]))
        train_classes = torch.tensor(np.unique(all_labels[train]))

        self.all_data = {
            'attributes_class_matrix': torch.tensor(attribute, dtype=torch.float32),
            'classes_names':           classes_names,
            'attribute_names':         attribute_names,
            'test_unseen_classes':     test_unseen_classes,
            'train_classes':           train_classes,
            'test_seen_classes':       train_classes,
            'all_classes':             torch.cat((train_classes, test_unseen_classes))
        }


    def train_dataloader(self):
        return DataLoader(self.ds_train, 
                          batch_size        =self.batch_size, 
                          shuffle           =True, 
                          num_workers       =self.num_workers,
                          persistent_workers=True)


    def val_dataloader(self):
        return [
            DataLoader(self.ds_test_zsl, 
                        batch_size        =self.batch_size, 
                        shuffle           =False, 
                        num_workers       =self.num_workers,
                        persistent_workers=True),
            DataLoader(self.ds_test_gzsl, 
                        batch_size        =self.batch_size, 
                        shuffle           =False, 
                        num_workers       =self.num_workers,
                        persistent_workers=True)
        ]


    def test_dataloader(self):
        return self.val_dataloader()


    def data_summary(self):
        print("\n--- Data Summary ---")
        print(f"Mode:                {'RAW IMAGES' if self.use_raw_images else 'FROZEN FEATURES'}")
        print(f"Attributes shape:    {self.all_data['attributes_class_matrix'].shape}")
        print(f"All classes:         {len(self.all_data['all_classes'])}")
        print(f"Test seen classes:   {len(self.all_data['test_seen_classes'])}")
        print(f"Test unseen classes: {len(self.all_data['test_unseen_classes'])}")
        print(f"Total attributes:    {len(self.all_data['attribute_names'])}")
        print("-" * 40, "\n")





# ╭───────────────────────────────────────────────────────────────────────────╮
# │                     Embedding Verification Helper                         │
# ╰───────────────────────────────────────────────────────────────────────────╯

def verify_embeddings_match(base_path='./data', dataset_name='AWA2', idx_to_check=1):
    print("\nVerifying Embeddings Match...")
    
    dataset_dir = os.path.join(base_path, 'xlsa17', 'data', dataset_name)
    res101 = scipy.io.loadmat(os.path.join(dataset_dir, 'res101.mat'))
    
    mat_feature = torch.tensor(np.transpose(res101['features'])[idx_to_check], dtype=torch.float32)
    
    raw_path_str = str(res101['image_files'].squeeze()[idx_to_check][0])
    
    # Same safe split logic here!
    relative_path = 'JPEGImages' + raw_path_str.split('JPEGImages')[-1]
    relative_path = relative_path.replace('//', '/')
    local_image_path = os.path.join(base_path, 'Animals_with_Attributes2', relative_path)
    
    if not os.path.exists(local_image_path):
        print(f"Could not find raw image at {local_image_path}. Did you download the raw images?")
        return
        
    resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    cnn = torch.nn.Sequential(*(list(resnet.children())[:-1]), torch.nn.Flatten())
    cnn.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(local_image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0) 

    with torch.no_grad():
        dynamic_feature = cnn(img_t).squeeze()

    cosine_similarity = torch.nn.functional.cosine_similarity
    cos_sim = cosine_similarity(dynamic_feature.unsqueeze(0), 
                                mat_feature.unsqueeze(0)).item()
    
    print(f"Testing Image: {relative_path}")
    print(f"Cosine Similarity between .mat feature and PyTorch feature: {cos_sim:.4f}")

    print(f"Original mean {mat_feature.mean()}")
    print(f"Original max  {mat_feature.max()}")
    print(f"Original min  {mat_feature.min()}")
    
    print(f"Local    mean {dynamic_feature.mean()}")
    print(f"Local    max  {dynamic_feature.max()}")
    print(f"Local    min  {dynamic_feature.min()}")

    if cos_sim > 0.90:
        print("✅ Match! The embeddings align beautifully.\n")
    else:
        print(f"⚠️ Similarity is low: {cos_sim}\n")



    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    plt.hist(
        mat_feature.detach().cpu().numpy().flatten(),
        bins=100,
        density=True,
        alpha=0.5,
        label="Original"
    )

    plt.hist(
        dynamic_feature.detach().cpu().numpy().flatten(),
        bins=100,
        density=True,
        alpha=0.5,
        label="Local"
    )

    plt.xlabel("Feature value")
    plt.ylabel("Density")
    plt.title("Feature Value Distribution")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()