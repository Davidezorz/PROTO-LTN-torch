import torch
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_tsne_with_synthetic_vectors(model, data_module, save_prefix="feature_tsnet"):
    print("Generating t-SNE visualization with synthetic vectors...")

    all_data = data_module.all_data
    
    # 1. Extract original attributes and class names
    attributes = all_data['attributes_class_matrix'].clone()
    class_names = all_data['classes_names'].copy()
    
    # Clean up class names as in the original script
    for f in range(len(class_names)):
        if "+" in class_names[f]:
            v = class_names[f].split("+")
            class_names[f] = v[0] + "+" + v[1][0] + "."

    # 2. Create synthetic vectors (Null and Full)
    num_attributes = attributes.shape[1]
    null_vector = torch.zeros((1, num_attributes), dtype=torch.float32)
    full_vector = torch.ones((1, num_attributes), dtype=torch.float32)
    mean_vector = attributes.float().mean(dim=0, keepdim=True)

    stripes_idx = all_data['attribute_names'].index('stripes')
    zebra_idx = all_data['classes_names'].index('zebra')
    print(f"stripes_idx: {stripes_idx}")
    # stripes = null_vector.clone()
    stripes = attributes[zebra_idx, :].clone()
    stripes[stripes_idx] = 0
    stripes = stripes[None, :]
    print(stripes)
    
    # 3. Concatenate and add labels
    combined_attributes = torch.cat([attributes, null_vector, full_vector, stripes, mean_vector], dim=0)
    class_names.extend(["all_zeros", "all_ones", "zebra_no_stripes", "mean_vector"])
    
    # 4. Forward pass through the embedding model to get prototypes
    model.eval()
    with torch.no_grad():
        combined_attributes = combined_attributes.to(model.device)
        # Using the EmbeddingModel to generate the 2048-D prototypes[cite: 5]
        prototypes = model.embeddingFunction(combined_attributes)
        prototypes_np = prototypes.cpu().numpy()

    # 5. Dimensionality Reduction
    # pca = PCA(n_components=50)
    # X_pca = pca.fit_transform(prototypes_np)
    X_pca = prototypes_np

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=42).fit_transform(X_pca)
    
    # 6. Prepare DataFrame for Seaborn
    df_subset = pd.DataFrame({
        'tsne-2d-one': tsne[:, 0],
        'tsne-2d-two': tsne[:, 1],
        'class': class_names
    })

    # Normalize coordinates
    df_subset['tsne-2d-one'] = (df_subset['tsne-2d-one'] - df_subset['tsne-2d-one'].min()) / (df_subset['tsne-2d-one'].max() - df_subset['tsne-2d-one'].min())
    df_subset['tsne-2d-two'] = (df_subset['tsne-2d-two'] - df_subset['tsne-2d-two'].min()) / (df_subset['tsne-2d-two'].max() - df_subset['tsne-2d-two'].min())

    # 7. Plotting
    plt.figure(figsize=(14, 10))
    sns.set_context('paper', font_scale=1)
    
    # Dynamically size the color palette based on the number of classes (50 original + 2 synthetic)
    palette = sns.color_palette("hls", len(class_names))
    
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="class",
        palette=palette,
        data=df_subset,
        legend=False,
        alpha=1, s=300
    )

    ax.set(xlabel=None, ylabel=None)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])

    # Custom label point function to prevent overlap
    def label_point(x, y, val):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        added = []
        for i, point in a.iterrows():
            px, py = point['x'], point['y']
            for j in added:
                if abs(j[0] - px) <= 0.05 and abs(j[1] - py) <= 0.05:
                    px -= 0.05
                    py -= 0.05
            added.append((px, py))
            
            # Make synthetic vectors bold and distinct for easier spotting
            red_names = ["all_zeros", "all_ones", "zebra_no_stripes", "mean_vector"]
            weight = 'bold' if point['val'] in red_names else 'normal'
            color = 'red' if point['val'] in red_names else 'black'
            
            plt.text(px, py, str(point['val']), horizontalalignment='center', weight=weight, color=color)

    label_point(df_subset['tsne-2d-one'], df_subset['tsne-2d-two'], df_subset['class'])

    # 8. Save and display
    plt.savefig(f"{save_prefix}_features.png")
    plt.savefig(f"{save_prefix}_features.pdf", bbox_inches='tight')
    # plt.show()
    print("Plot saved successfully.")





def plot_image_embeddings_tsne(model, data_module, samples_per_class=10, save_prefix="image_tsne"):
    print(f"Collecting {samples_per_class} images per class for t-SNE...")

    # Access the Generalized ZSL dataset and class names[cite: 3]
    dataset = data_module.ds_test_gzsl 
    all_data = data_module.all_data
    class_names = all_data['classes_names'].copy()
    
    # Clean up class names
    for f in range(len(class_names)):
        if "+" in class_names[f]:
            v = class_names[f].split("+")
            class_names[f] = v[0] + "+" + v[1][0] + "."

    collected_features = []
    collected_labels = []
    marker_styles = [] # To differentiate images from prototypes
    
    class_counts = {i: 0 for i in range(len(class_names))}
    
    model.eval()
    with torch.no_grad():
        # 1. Collect Image Features
        for i in range(len(dataset)):
            data, label, _ = dataset[i]
            label_idx = label.item()
            
            if class_counts[label_idx] < samples_per_class:
                # Add batch dimension and move to device
                data = data.unsqueeze(0).to(model.device)
                
                # Compute visual features (raw CNN or frozen)[cite: 4]
                if model.compute_feature:
                    feature = model.cnn(data)
                else:
                    feature = data
                    
                collected_features.append(feature.cpu().squeeze().numpy())
                collected_labels.append(class_names[label_idx])
                marker_styles.append("Image")
                class_counts[label_idx] += 1
            
            # Stop if we have enough samples for all classes
            if all(count >= samples_per_class for count in class_counts.values()):
                break
                
        # 2. Collect Class Prototypes
        print("Computing class prototypes...")
        attributes = all_data['attributes_class_matrix'].to(model.device)
        prototypes = model.embeddingFunction(attributes)
        
        for i in range(len(class_names)):
            collected_features.append(prototypes[i].cpu().squeeze().numpy())
            collected_labels.append(class_names[i])
            marker_styles.append("Prototype")

    # 3. Dimensionality Reduction
    print("Running t-SNE...")
    features_np = np.array(collected_features)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=42).fit_transform(features_np)
    
    # 4. Prepare DataFrame
    df_subset = pd.DataFrame({
        'tsne-2d-one': tsne[:, 0],
        'tsne-2d-two': tsne[:, 1],
        'Class': collected_labels,
        'Type': marker_styles
    })

    # Normalize coordinates
    df_subset['tsne-2d-one'] = (df_subset['tsne-2d-one'] - df_subset['tsne-2d-one'].min()) / (df_subset['tsne-2d-one'].max() - df_subset['tsne-2d-one'].min())
    df_subset['tsne-2d-two'] = (df_subset['tsne-2d-two'] - df_subset['tsne-2d-two'].min()) / (df_subset['tsne-2d-two'].max() - df_subset['tsne-2d-two'].min())

    # 5. Plotting
    plt.figure(figsize=(16, 12))
    sns.set_context('paper', font_scale=1.2)
    
    palette = sns.color_palette("hls", len(class_names))
    
    # Plot using different markers for Images (circles) and Prototypes (stars or larger circles)
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Class",
        style="Type",
        palette=palette,
        data=df_subset,
        markers={"Image": "o", "Prototype": "X"},
        s=150, # Base size
        alpha=0.8
    )

    ax.set(xlabel=None, ylabel=None)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2)

    plt.savefig(f"{save_prefix}.png", bbox_inches='tight')
    plt.savefig(f"{save_prefix}.pdf", bbox_inches='tight')
    print("Image embedding plot saved successfully.")




def plot_image_embeddings_with_centroids(model, data_module, samples_per_class=7, save_prefix="image_tsne"):
    print(f"Collecting {samples_per_class} images per class for t-SNE...")

    dataset = data_module.ds_test_gzsl 
    all_data = data_module.all_data
    class_names = all_data['classes_names'].copy()
    
    # Clean up class names
    for f in range(len(class_names)):
        if "+" in class_names[f]:
            v = class_names[f].split("+")
            class_names[f] = v[0] + "+" + v[1][0] + "."

    collected_features = []
    collected_labels = []
    marker_styles = [] # "Image" or "Prototype"
    
    class_counts = {i: 0 for i in range(len(class_names))}
    
    model.eval()
    with torch.no_grad():
        # 1. Collect Image Features (Computed by the CNN)[cite: 4]
        for i in range(len(dataset)):
            data, label, _ = dataset[i]
            label_idx = label.item()
            
            if class_counts[label_idx] < samples_per_class:
                data = data.unsqueeze(0).to(model.device)
                
                if model.compute_feature:
                    feature = model.cnn(data)
                else:
                    feature = data
                    
                collected_features.append(feature.cpu().squeeze().numpy())
                collected_labels.append(class_names[label_idx])
                marker_styles.append("Image")
                class_counts[label_idx] += 1
            
            if all(count >= samples_per_class for count in class_counts.values()):
                break
                
        # 2. Collect Class Prototypes (Computed by the Embedding NN)[cite: 5]
        print("Computing class prototypes (Semantic Embeddings)...")
        attributes = all_data['attributes_class_matrix'].to(model.device)
        prototypes = model.embeddingFunction(attributes)
        
        for i in range(len(class_names)):
            collected_features.append(prototypes[i].cpu().squeeze().numpy())
            collected_labels.append(class_names[i])
            marker_styles.append("Prototype")

    # 3. Dimensionality Reduction
    print("Running t-SNE...")
    features_np = np.array(collected_features)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=42).fit_transform(features_np)
    
    # 4. Prepare DataFrame
    df_subset = pd.DataFrame({
        'tsne-2d-one': tsne[:, 0],
        'tsne-2d-two': tsne[:, 1],
        'Class': collected_labels,
        'Type': marker_styles
    })

    # Normalize coordinates
    df_subset['tsne-2d-one'] = (df_subset['tsne-2d-one'] - df_subset['tsne-2d-one'].min()) / (df_subset['tsne-2d-one'].max() - df_subset['tsne-2d-one'].min())
    df_subset['tsne-2d-two'] = (df_subset['tsne-2d-two'] - df_subset['tsne-2d-two'].min()) / (df_subset['tsne-2d-two'].max() - df_subset['tsne-2d-two'].min())

    # 5. Plotting
    plt.figure(figsize=(18, 14))
    sns.set_context('paper', font_scale=1.2)
    
    palette = sns.color_palette("hls", len(class_names))
    
    # Plot the images as standard circles
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Class",
        style="Type",
        palette=palette,
        data=df_subset[df_subset['Type'] == 'Image'],
        markers={"Image": "o"},
        s=100, 
        alpha=0.6,
        legend=False
    )

    # Plot the prototypes (Embedding NN outputs) as large stars
    prototype_df = df_subset[df_subset['Type'] == 'Prototype']
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Class",
        palette=palette,
        data=prototype_df,
        marker="*",
        s=800, # Make the centroid highly visible
        edgecolor="black",
        legend=False,
        ax=ax
    )

    ax.set(xlabel=None, ylabel=None)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])

    # 6. Add labels directly on the centroids (prototypes)
    for _, row in prototype_df.iterrows():
        plt.text(
            row['tsne-2d-one'], 
            row['tsne-2d-two'] + 0.012, # Slight Y offset so it sits above the star
            row['Class'], 
            horizontalalignment='center', 
            weight='bold', 
            color='black',
            fontsize=10
        )

    plt.savefig(f"{save_prefix}_centroids.png", bbox_inches='tight')
    plt.savefig(f"{save_prefix}_centroids.pdf", bbox_inches='tight')
    print("Image embedding plot with labeled centroids saved successfully.")





def plot_image_embeddings_with_actual_centroids(model, data_module, samples_per_class=10, save_prefix="image_tsne"):
    print(f"Collecting {samples_per_class} images per class for t-SNE...")

    dataset = data_module.ds_test_gzsl 
    all_data = data_module.all_data
    class_names = all_data['classes_names'].copy()
    
    # Clean up class names
    for f in range(len(class_names)):
        if "+" in class_names[f]:
            v = class_names[f].split("+")
            class_names[f] = v[0] + "+" + v[1][0] + "."

    collected_features = []
    collected_labels = []
    marker_styles = [] 
    
    class_counts = {i: 0 for i in range(len(class_names))}
    features_per_class = {name: [] for name in class_names}
    
    model.eval()
    with torch.no_grad():
        # 1. Collect Image Features
        for i in range(len(dataset)):
            data, label, _ = dataset[i]
            label_idx = label.item()
            
            if class_counts[label_idx] < samples_per_class:
                data = data.unsqueeze(0).to(model.device)
                
                if model.compute_feature:
                    feature = model.cnn(data)
                else:
                    feature = data
                    
                feat_np = feature.cpu().squeeze().numpy()
                
                collected_features.append(feat_np)
                collected_labels.append(class_names[label_idx])
                marker_styles.append("Image")
                
                features_per_class[class_names[label_idx]].append(feat_np)
                class_counts[label_idx] += 1
            
            if all(count >= samples_per_class for count in class_counts.values()):
                break
                
        # 2. Compute Actual Centroids 
        print("Computing actual image centroids...")
        for name in class_names:
            if len(features_per_class[name]) > 0:
                class_mean = np.mean(features_per_class[name], axis=0)
                collected_features.append(class_mean)
                collected_labels.append(name)
                marker_styles.append("Actual Centroid")

        # 3. Collect Class Prototypes 
        print("Computing class prototypes...")
        attributes = all_data['attributes_class_matrix'].to(model.device)
        prototypes = model.embeddingFunction(attributes)
        
        for i in range(len(class_names)):
            collected_features.append(prototypes[i].cpu().squeeze().numpy())
            collected_labels.append(class_names[i])
            marker_styles.append("Prototype")

    # 4. Dimensionality Reduction
    print("Running t-SNE...")
    features_np = np.array(collected_features)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=42).fit_transform(features_np)
    
    df_subset = pd.DataFrame({
        'tsne-2d-one': tsne[:, 0],
        'tsne-2d-two': tsne[:, 1],
        'Class': collected_labels,
        'Type': marker_styles
    })

    df_subset['tsne-2d-one'] = (df_subset['tsne-2d-one'] - df_subset['tsne-2d-one'].min()) / (df_subset['tsne-2d-one'].max() - df_subset['tsne-2d-one'].min())
    df_subset['tsne-2d-two'] = (df_subset['tsne-2d-two'] - df_subset['tsne-2d-two'].min()) / (df_subset['tsne-2d-two'].max() - df_subset['tsne-2d-two'].min())

    # --- FIX 1: EXPLICIT COLOR MAPPING ---
    # Create a fixed dictionary mapping each class to a specific color
    palette_colors = sns.color_palette("hls", len(class_names))
    color_map = dict(zip(class_names, palette_colors))

    plt.figure(figsize=(20, 16))
    sns.set_context('paper', font_scale=1.2)
    
    # Plot Images
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Class",
        palette=color_map, # Use the explicit map
        data=df_subset[df_subset['Type'] == 'Image'],
        marker="o",
        s=80, 
        alpha=0.4, # Made slightly more transparent to help text pop
        legend=False
    )

    # Plot Actual Centroids
    actual_centroid_df = df_subset[df_subset['Type'] == 'Actual Centroid']
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Class",
        palette=color_map, # Use the explicit map
        data=actual_centroid_df,
        marker="D", 
        s=250, 
        edgecolor="black",
        linewidth=1.5,
        legend=False,
        ax=ax
    )

    # Plot Prototypes
    prototype_df = df_subset[df_subset['Type'] == 'Prototype']
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Class",
        palette=color_map, # Use the explicit map
        data=prototype_df,
        marker="*",
        s=700, 
        edgecolor="black",
        linewidth=1.5,
        legend=False,
        ax=ax
    )

    ax.set(xlabel=None, ylabel=None)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])

    # --- FIX 2: COLORED LABELS WITH ANTI-OVERLAP AND BACKGROUND ---
    placed_text_positions = []
    
    for _, row in prototype_df.iterrows():
        x, y = row['tsne-2d-one'], row['tsne-2d-two']
        
        # Simple nudge if too close to an already placed text
        for px, py in placed_text_positions:
            if abs(x - px) < 0.025 and abs(y - py) < 0.025:
                y += 0.025 # Nudge up
                x += 0.015 # Nudge right
                
        placed_text_positions.append((x, y))
        
        plt.text(
            x, 
            y + 0.015, 
            row['Class'], 
            horizontalalignment='center', 
            weight='bold', 
            color=color_map[row['Class']], # Text color matches the class
            fontsize=8,
            # Add a white box behind the text so it covers the messy scatter points
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1) 
        )

    plt.savefig(f"{save_prefix}_comparison.png", bbox_inches='tight', dpi=300) # Added dpi for clarity
    plt.savefig(f"{save_prefix}_comparison.pdf", bbox_inches='tight')
    print("Plot with corrected colors and improved labels saved successfully.")