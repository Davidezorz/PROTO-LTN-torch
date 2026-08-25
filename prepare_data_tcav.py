import os
import glob
import random
import shutil

def prepare_tcav_concepts():
    # --- Paths ---
    broden_dir = "./data/broden1_224/images/dtd/"
    awa2_dir = "./data/Animals_with_Attributes2/JPEGImages/"
    
    # Notice the extra 'images/' subfolder - this is required by PyTorch ImageFolder!
    stripes_target = "./data/concepts/stripes/images/"
    random_target = "./data/concepts/random/images/"

    # --- 1. Process STRIPES ---
    print("Preparing 'stripes' concept...")
    if os.path.exists("./data/concepts/stripes/"):
        shutil.rmtree("./data/concepts/stripes/")
    os.makedirs(stripes_target)

    # glob naturally filters out the _color.png files by strictly looking for .jpg
    striped_paths = glob.glob(os.path.join(broden_dir, "*striped*.jpg"))
    
    # Optional: Filter strictly for 'striped_XXXX.jpg' if needed
    striped_paths = [p for p in striped_paths if "_color" not in p]

    for src_path in striped_paths:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(stripes_target, filename)
        shutil.copy(src_path, dst_path)
        
    print(f"✅ Copied {len(striped_paths)} striped images.")

    # --- 2. Process RANDOM ---
    print("\nPreparing 'random' concept...")
    if os.path.exists("./data/concepts/random/"):
        shutil.rmtree("./data/concepts/random/")
    os.makedirs(random_target)

    all_awa2_paths = []
    for root, dirs, files in os.walk(awa2_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                all_awa2_paths.append(os.path.join(root, file))

    # Match the number of random images to the number of stripe images
    num_random = len(striped_paths) if len(striped_paths) > 0 else 120
    
    random.seed(42)
    random.shuffle(all_awa2_paths)
    selected_random_paths = all_awa2_paths[:num_random]

    for i, src_path in enumerate(selected_random_paths):
        dst_path = os.path.join(random_target, f"random_{i:03d}.jpg")
        shutil.copy(src_path, dst_path)

    print(f"✅ Copied {len(selected_random_paths)} random images.")
    print("\nConcept folders are ready for Captum!")

if __name__ == "__main__":
    prepare_tcav_concepts()