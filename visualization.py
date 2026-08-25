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

    stripes_idx = all_data['attribute_names'].index('stripes')
    zebra_idx = all_data['classes_names'].index('zebra')
    print(f"stripes_idx: {stripes_idx}")
    # stripes = null_vector.clone()
    stripes = attributes[zebra_idx, :].clone()
    stripes[stripes_idx] = 0
    stripes = stripes[None, :]
    print(stripes)
    
    # 3. Concatenate and add labels
    combined_attributes = torch.cat([attributes, null_vector, full_vector, stripes], dim=0)
    class_names.extend(["all_zeros", "all_ones", "stripes"])
    
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
            red_names = ["all_zeros", "all_ones", "stripes"]
            weight = 'bold' if point['val'] in red_names else 'normal'
            color = 'red' if point['val'] in red_names else 'black'
            
            plt.text(px, py, str(point['val']), horizontalalignment='center', weight=weight, color=color)

    label_point(df_subset['tsne-2d-one'], df_subset['tsne-2d-two'], df_subset['class'])

    # 8. Save and display
    plt.savefig(f"{save_prefix}_features.png")
    plt.savefig(f"{save_prefix}_features.pdf", bbox_inches='tight')
    plt.show()
    print("Plot saved successfully.")