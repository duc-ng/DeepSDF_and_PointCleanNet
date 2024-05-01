import trimesh
import numpy as np
import numpy as np
from scipy.spatial import KDTree
from src.AutoDecoder import AutoDecoder
from src.dataset import DeepSDF_Dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import metrics
import os
from PIL import Image
from glob import glob

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def plot_confusion_matrix(train_loader, model, latent):
    y_true = []
    y_pred = []    
    # iterate over the dataset
    
    for i, (x, y, indices) in enumerate(train_loader):
        # get data
        inputs = x.to(device)
        latent_vector = latent[indices]
        
        # add latent vector to input coords and reshape
        inputs = torch.cat(
            (inputs, latent_vector.unsqueeze(1).repeat(1, inputs.shape[1], 1)),
            dim=2,
        )
        inputs = inputs.view(-1, inputs.shape[2])
        targets = y.view(-1, 1).squeeze().cpu().detach().numpy()

        # predict
        outputs = model(inputs).squeeze().cpu().detach().numpy()
        
        # map values to -1, 1 and 0 if close
        targets = np.sign(targets)
        outputs = [0 if abs(x) < 0.001 else np.sign(x) for x in outputs]

        # concat
        y_true = y_true + targets.tolist()
        y_pred = y_pred + list(outputs)

    cm=metrics.confusion_matrix(y_true,y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels = [-1, 0, 1])
    cm_display.plot()
    plt.savefig("report/confusion_matrix.png")
    print("Confusion matrix saved in report/confusion_matrix.png")

def plot_iterations(directory):
    """
    Plot the 20 images in a grid.
    """
    image_files = [os.path.join(directory, str(f)+".png") for f in range(0, 25)]
    
    # Labels 
    labels = range(500, 2001, 500)
    _, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 14), dpi=100)
    column_labels = ["Original"]+[f"Iterations={label}" for label in labels]
    row_labels = [f"Bunny {i+1}" for i in range(5)]
    
    # Plot each image
    for ax, image_path in zip(axes.flat, image_files):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.set_xticks([])  
        ax.set_yticks([]) 
        for spine in ax.spines.values():  
            spine.set_visible(False)
        
    # Set column labels
    for ax, col_label in zip(axes[-1], column_labels):
        ax.set_xlabel(col_label, fontdict={'fontsize': 12})
        
    # row labels
    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.annotate(row_label, xy=(-0.1, 0.5), xycoords="axes fraction", 
                    textcoords="offset points", va="center", ha="right", fontsize=12)

    # Overall title and layout adjustment
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.08)
    plt.savefig('report/iterations.png', dpi=150)

def plot_sdf(directory):
    image_files = [os.path.join(directory, f"{i}.png") for i in range(4)]
    labels = ["All", "Boundary", "Inside", "Outside"]
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))

    for ax, image_path, label in zip(axes, image_files, labels):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.set_xlabel(label, fontdict={'fontsize': 12})
        ax.set_xticks([])  # Remove x-axis tick marks
        ax.set_yticks([])  # Remove y-axis tick marks
        for spine in ax.spines.values():  # Remove the borders of the plot
            spine.set_visible(False)

    plt.tight_layout(pad=0)
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('report/sdf.png', dpi=150)

def plot_noise(directory):
    """
    Plot the 3x3 images in a grid.
    """
    image_files = [os.path.join(directory, str(f)+".png") for f in range(9)]
    
    # Labels 
    _, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 14), dpi=100)
    column_labels = ["Original", "Input PCD", "Reconstruction"] 
    row_labels = ["No Noise", "Noise 0.03", "Noise 0.05 and PCNet"]
    
    # Plot each image
    for ax, image_path in zip(axes.flat, image_files):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.set_xticks([])  
        ax.set_yticks([]) 
        for spine in ax.spines.values():  
            spine.set_visible(False)
        
    # Set column labels
    for ax, col_label in zip(axes[-1], column_labels):
        ax.set_xlabel(col_label, fontdict={'fontsize': 12})
        
    # row labels
    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.annotate(row_label, xy=(-0.1, 0.5), xycoords="axes fraction", 
                    textcoords="offset points", va="center", ha="right", fontsize=12)

    # Overall title and layout adjustment
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.17)
    plt.savefig('report/noise.png', dpi=150)


########################################## init ############################################################
device = torch.device("mps")
nr_chamfer_samples = 30000
batch_size = 5
epochs = 2000
weights = f"weights/model_{epochs}.pth"
latent_size = 128
nr_rand_samples = 10000
latent_file = "weights/latent.pt"
############################################################################################################


if __name__ == "__main__":
    
    # compute Chamfer Distance (Reconstruction)
    mesh_files = glob(os.path.join("out/1_preprocessed", "*.obj"))
    basenames = [os.path.basename(f).split(".")[0] for f in mesh_files]
    meshes = [trimesh.load(f) for f in mesh_files]
    meshes_reconstructed = [trimesh.load(f"out/2_reconstructed/{b}/{epochs}.obj") for b in basenames]
    points = [trimesh.sample.sample_surface(m, nr_chamfer_samples)[0] for m in meshes]
    points2 = [trimesh.sample.sample_surface(m, nr_chamfer_samples)[0] for m in meshes_reconstructed]
    distances = [chamfer_distance(points[i], points2[i]) for i in range(len(points))]
    print("Mean Chamfer Distance (Reconstruction):", np.mean(distances))
    
    # compute Chamfer Distance (Shape Completion)
    mesh_files = sorted(glob(os.path.join("out/shape_completion", "*.obj")))
    basenames = [int(os.path.basename(f).split(".")[0]) for f in mesh_files]
    mesh_files = [f for _, f in sorted(zip(basenames, mesh_files))]
    mesh = trimesh.load("out/1_preprocessed/bunny.obj")
    meshes_reconstructed = [trimesh.load(mesh_file) for mesh_file in mesh_files]
    points = trimesh.sample.sample_surface(mesh, nr_chamfer_samples)[0]
    points2 = [trimesh.sample.sample_surface(m, nr_chamfer_samples)[0] for m in meshes_reconstructed]
    distances = [chamfer_distance(points, p) for p in points2]
    distances = [round(d, 4) for d in distances]
    print("Mean Chamfer Distance (shape completion):", distances)


    # plot confusion matrix
    model = AutoDecoder(latent_size).float().eval().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    latent = torch.load(latent_file, map_location=device).weight
    train_dataset = DeepSDF_Dataset(nr_rand_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    plot_confusion_matrix(train_loader, model, latent)
    
    # plot iterations
    plot_iterations("report/iterations")
    
    # plot sdf
    plot_sdf("report/sdf")
    
    # plot noise
    plot_noise("report/noise")