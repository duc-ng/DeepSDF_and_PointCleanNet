import argparse
import trimesh
import numpy as np
import os
import torch
from src.AutoDecoder import AutoDecoder
from tqdm import tqdm
from src.dataset import SingleShape_Dataset
from train import loss_function, save_plot_losses
from predict import save_reconstructions
import subprocess


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description="DeepSDF and PointCleanNet")

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="device to train on")
    parser.add_argument(
        "--input_mesh",
        type=str,
        default="out/1_preprocessed/bunny.obj",
        help="input mesh",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/shape_completion",
        help="output directory",
    )
    parser.add_argument(
        "--number_of_samples",
        type=int,
        default=20000,
        help="number of samples on surface",
    )
    parser.add_argument("--latent_size", type=int, default=128, help="latent size")
    parser.add_argument(
        "--latent_lr", type=float, default=0.00002, help="learning rate"
    )
    parser.add_argument("--latent_std", type=float, default=0.01, help="latent std")
    parser.add_argument("--latent_epochs", type=int, default=99, help="latent epochs")
    parser.add_argument(
        "--delta", type=float, default=0.1, help="delta for clamping loss function"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="batch size for training"
    )
    parser.add_argument(
        "--batch_size_reconstruct",
        type=int,
        default=100000,
        help="batch size for reconstruction",
    )
    parser.add_argument("--N", type=int, default=192, help="meshgrid size")
    parser.add_argument(
        "--weight_file", type=str, default="weights/model_1500.pth", help="weight file"
    )
    parser.add_argument("--add_noise", action="store_true", help="add noise to the points")
    parser.add_argument("--noise_std", type=float, default=0.01, help="noise std")

    parser.add_argument(
        "--clean",
        action="store_true",
        help="clean the noise with PointCleanNet",
    )
    parser.add_argument(
        "--clean_nrun",
        type=int,
        default=2,
        help="Number of runs with PointCleanNet noise removal",
    )
    return parser.parse_args()


def estimate_latent(
    model,
    dataloader,
    latent_inference,
    lr_latent,
    latent_epochs,
    delta,
    latent_std,
    output_dir,
):
    optimizer = torch.optim.Adam([latent_inference.weight], lr=lr_latent)
    epoch_losses = []
    latents = []
    epochs = []

    for epoch in range(latent_epochs):
        losses = []

        with tqdm(dataloader, unit="batch") as tepoch:
            for x, y in tepoch:

                # get inputs and targets
                tepoch.set_description(f"Epoch {epoch+1}/{latent_epochs}")

                # move data to device
                inputs = x.to(device)
                targets = y.to(device)
                latent_vector = latent_inference.weight.to(device)

                # add latent vector to input coords
                inputs = torch.cat(
                    (inputs, latent_vector.repeat(inputs.shape[0], 1)), dim=1
                )

                # predict
                optimizer.zero_grad()
                outputs = model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets, delta, latent_vector, latent_std)
                losses.append(loss.item())

                # backprop
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=np.mean(losses))

        epoch_losses.append(np.mean(losses))

        # save latent vector
        if (epoch + 1) % 20 == 0:
            latents.append(latent_inference.weight.detach().clone())
            epochs.append(epoch + 1)

    save_plot_losses(epoch_losses, output_dir)

    return latents, epochs


if __name__ == "__main__":

    # init
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    mesh = trimesh.load(args.input_mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)

    # sample points on surface
    samples, faces = trimesh.sample.sample_surface(mesh, args.number_of_samples)
    normals = mesh.face_normals[faces]
    delta = 0.01
    positive_points = samples + delta * normals
    negative_points = samples - delta * normals
    points = np.vstack((positive_points, negative_points))
    print(f"Number of sampled points: {len(points)}")

    # targets is delta for each positive_points
    targets_pos = delta * np.ones(len(positive_points), dtype=np.float32)
    targets_neg = -delta * np.ones(len(negative_points), dtype=np.float32)
    targets = np.hstack((targets_pos, targets_neg))

    # remove points smaller than 0 on z-axis
    mask = points[:, 2] > 0
    points = points[mask]
    targets = targets[mask]
    print(f"Number of partial points: {len(points)}")
    
    # add noise
    if args.add_noise:
        noise = np.random.normal(0, args.noise_std, points.shape)
        points += noise

    # save points as .npy, .obj, .xyz
    points_directory = os.path.join(args.output_dir, "points")
    os.makedirs(points_directory, exist_ok=True)
    np.save(os.path.join(points_directory, "points.npy"), points)
    mesh_out = trimesh.Trimesh(vertices=points)
    mesh_out.export(os.path.join(points_directory, "points.obj"))
    with open(os.path.join(points_directory, "points.xyz"), "w") as f:
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    
    # run PointCleanNet and load cleaned points
    if args.clean:
        command = [
            "python",
            "src/pointcleannet.py",
            "--input_dir",
            points_directory,
            "--outlier_dir",
            os.path.join(args.output_dir, "pcnet", "outliers"),
            "--output_dir",
            os.path.join(args.output_dir, "pcnet", "cleaned_noise"),
            "--device",
            str(device),
            "--seed",
            str(args.seed),
            "--nrun",
            str(args.clean_nrun),
        ]
        subprocess.run(command)
        
        info = np.loadtxt(os.path.join(args.output_dir, "pcnet", "outliers", "points.info"))
        targets = targets[info[:, 3] > 0.5] # remove outliers from targets
        points = np.loadtxt(os.path.join(args.output_dir, "pcnet", "cleaned_noise", f"points_{args.clean_nrun}.xyz"))

    # load data and model
    dataset = SingleShape_Dataset(points, targets)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    model = AutoDecoder(latent_size=args.latent_size).float().train().to(device)
    model.load_state_dict(torch.load(args.weight_file, map_location=device))

    # estimate latent vectors for inference
    latent_inference = torch.nn.Embedding(1, args.latent_size).to(device)
    torch.nn.init.normal_(latent_inference.weight.data, mean=0.0, std=args.latent_std)

    latents, epochs = estimate_latent(
        model,
        dataloader,
        latent_inference,
        args.latent_lr,
        args.latent_epochs,
        args.delta,
        args.latent_std,
        args.output_dir,
    )

    # reconstruct shape
    model.eval()
    for latent, epoch in zip(latents, epochs):
        output_file = os.path.join(args.output_dir, f"{epoch}.obj")
        save_reconstructions(
            model, args.batch_size_reconstruct, device, args.N, latent, output_file
        )

   
