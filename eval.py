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

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def plot_confusion_matrix(train_loader, model):
    y = []
    y_pred = []
    for i, data in enumerate(train_loader):
        # get data
        input = data[:, 0:3].to(device)
        target = data[:, 3].numpy()
        output = model(input).detach().cpu().numpy().squeeze()
        
        # map values to -1, 1 and 0 if close
        target = np.sign(target)
        output = [0 if abs(x) < 0.001 else np.sign(x) for x in output]

        # append
        y = y + list(target)
        y_pred = y_pred + list(output)
        
    cm=metrics.confusion_matrix(y,y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels = [-1, 0, 1])
    cm_display.plot()
    os.makedirs("out/confusion_matrix", exist_ok=True)
    plt.savefig("out/confusion_matrix/confusion_matrix.png")
    print("Confusion matrix saved in out/confusion_matrix/confusion_matrix.png")
    

############################################################################################################
# init
nr_samples = 30000
nr_runs = 2
nr_epochs = 200
device = "mps"
name = "bunny"
batch_size = 6400
############################################################################################################


if __name__ == "__main__":
    
    # load meshes and sample points
    mesh_files = [
        f"out/1_preprocessed/{name}.obj",
        f"out/2_reconstructed/{name}.obj",
        f"out/3_cleaned_outliers/{name}.obj",
        f"out/4_cleaned_noise/{name}_{nr_runs}.obj",
    ]
    meshes = [trimesh.load(f) for f in mesh_files]
    points = [trimesh.sample.sample_surface(m, nr_samples)[0] for m in meshes]

    # compute the Chamfer Distance
    distances = [chamfer_distance(points[0], points[i]) for i in range(1, len(points))]
    print("Chamfer Distance (original and reconstructed):", distances[0])
    print("Chamfer Distance (original and cleaned outliers):", distances[1])
    print("Chamfer Distance (original and cleaned noise):", distances[2])

    # init model and dataset
    model = AutoDecoder().float().eval().to(device)
    model.load_state_dict(torch.load(f"out/2_reconstructed/bunny_{nr_epochs}.pth"))
    train_dataset = DeepSDF_Dataset(f"out/1_preprocessed/{name}.npz")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    # plot confusion matrix
    plot_confusion_matrix(train_loader, model)
        

    