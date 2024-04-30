import argparse
import trimesh
import numpy as np
import os
import torch
from src.AutoDecoder import AutoDecoder
from tqdm import tqdm
from src.dataset import SingleShape_Dataset
from train import loss_function
from predict import save_reconstructions


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
        "--number_of_samples", type=int, default=10000, help="number of samples"
    )
    parser.add_argument("--latent_size", type=int, default=128, help="latent size")
    parser.add_argument("--latent_lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--latent_std", type=float, default=0.01, help="latent std")
    parser.add_argument("--latent_epochs", type=int, default=50, help="latent epochs")
    parser.add_argument(
        "--delta", type=float, default=0.1, help="delta for clamping loss function"
    )
    parser.add_argument("--batch_size", type=int, default=6400, help="batch size")
    parser.add_argument(
        "--batch_size_reconstruct", type=int, default=100000, help="batch size"
    )
    parser.add_argument("--N", type=int, default=150, help="meshgrid size")
    
    # for PointCleanNet
    # parser.add_argument(
    #     "--clean",
    #     action="store_true",
    #     help="clean the reconstruction with PointCleanNet",
    # )
    # parser.add_argument(
    #     "--clean_nrun",
    #     type=int,
    #     default=2,
    #     help="Number of runs with PointCleanNet noise removal",
    # )
    return parser.parse_args()


def estimate_latent(
    model, dataloader, latent_inference, lr_latent, latent_epochs, delta, latent_std
):
    optimizer = torch.optim.Adam([latent_inference.weight], lr=lr_latent)

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

    return latent_inference.weight


if __name__ == "__main__":

    # init
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    mesh = trimesh.load(args.input_mesh)

    # sample points on surface
    points, _ = trimesh.sample.sample_surface(mesh, args.number_of_samples)
    print(f"Number of sampled points: {len(points)}")

    # remove points smaller than 0 on z-axis
    points = points[points[:, 2] > 0]
    print(f"Number of partial points: {len(points)}")

    # save points
    mesh = trimesh.Trimesh(vertices=points)
    mesh.export(os.path.join(args.output_dir, "points.obj"))

    # load data and model
    dataset = SingleShape_Dataset(mesh.vertices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    model = AutoDecoder(latent_size=args.latent_size).float().train().to(device)

    # estimate latent vectors for inference
    latent_inference = torch.nn.Embedding(1, args.latent_size).to(device)
    torch.nn.init.normal_(latent_inference.weight.data, mean=0.0, std=args.latent_std)
    latent_inference = estimate_latent(
        model,
        dataloader,
        latent_inference,
        args.latent_lr,
        args.latent_epochs,
        args.delta,
        args.latent_std,
    )
    
    # reconstruct shape
    output_file = os.path.join(args.output_dir, "shape_completion")
    save_reconstructions(
        model,
        args.batch_size_reconstruct,
        device,
        args.N,
        latent_inference,
        output_file
    )

    # # PointCleanNet
    # if args.clean:
    #     command = [
    #         "python",
    #         "src/pointcleannet.py",
    #         "--input_dir",
    #         args.output_dir,
    #         "--outlier_dir",
    #         "out/3_cleaned_outliers",
    #         "--output_dir",
    #         "out/4_cleaned_noise",
    #         "--device",
    #         str(device),
    #         "--seed",
    #         str(args.seed),
    #         "--nrun",
    #         str(args.clean_nrun),
    #     ]
    #     subprocess.run(command)
