# DeepSDF and PointCleanNet

This project constitutes Coursework 3 within the curriculum of COMP0119 - Acquisition and Processing of 3D Geometry (UCL). We have reconstructed the network presented in [DeepSDF](https://arxiv.org/abs/1901.05103), and have trained it on different meshes belonging to a `bunny` class. We have further expanded upon this by integrating [PointCleanNet](https://geometry.cs.ucl.ac.uk/projects/2019/pointcleannet/) into our methodology for mesh reconstruction. Comprehensive details can be found in the submitted reports.

## Installation

Set up the project with the following commands:

``` bash
conda create -n deepsdf python=3.10 -y
conda activate deepsdf
pip install torch torchvision torchaudio open3d trimesh rtree scikit-image
```

Then download the weights of PointCleanNet:

``` bash
cd pointcleannet/models && python download_models.py --task denoising && python download_models.py --task outliers_removal && cd ../..
```

## Usage

Start with preprocessing your `.obj` data files by running:
``` bash
python preprocess.py
```
| Argument               | Description                                      | Default Value         |
|------------------------|--------------------------------------------------|-----------------------|
| `--input_dir`          | Directory containing the `.obj` files to process | `data`                |
| `--output_dir`         | Directory where the output `.npz` files will be saved | `out/1_preprocessed` |
| `--samples_on_surface` | Number of samples to take directly on the surface of the mesh | `10000`              |
| `--samples_in_bbox`    | Number of samples to take within the bounding box of the mesh | `10000`              |
| `--samples_in_cube`    | Number of samples to take within the unit cube | `5000`                  |



You can train the SDF model with:
``` bash
python train.py --device cuda
```
| Argument        | Description                                       | Default Value                     |
|-----------------|---------------------------------------------------|-----------------------------------|
| `--weights_dir` | Directory for saving model weights                      | `out/weights`    |
| `--seed`        | Random seed                                       | `42`                              |
| `--device`      | Device to train on                                | `cpu`                             |
| `--batch-size`  | Input batch size for training                     | `5`                           |
| `--num-workers` | Number of workers                                 | `4`                               |
| `--lr_model`    | Learning rate for model training                  | `0.0001`                           |
| `--lr_latent`   | Learning rate for training latent vector          | `0.001`                           |
| `--epochs`      | Number of epochs to train                         | `2000`                             |
| `--delta`       | Delta for clamping                                | `0.1`                             |
| `--latent_size` | Size of latent vector                              | `128`                             |
| `--latent_std`       | Sigma value for initializing latent vector    | `0.01`                           |
| `--nr_rand_samples`  | Number of random subsamples from each batch   | `10000`                               |


And reconstruct the meshes with the trained latent vectors:
``` bash
python predict.py --device cuda
```
| Argument                | Description                                  | Default Value              |
|-------------------------|----------------------------------------------|----------------------------|
| `--input_dir`           | Input directory                              | `out/1_preprocessed`       |
| `--output_dir`          | Output directory                             | `out/2_reconstructed`      |
| `--weights_dir`         | Model weights directory                      | `out/weights`              |
| `--seed`                | Random seed                                  | `42`                       |
| `--device`              | Device to run on                             | `cpu`                      |
| `--num-workers`         | Number of workers                            | `4`                        |
| `--N`                   | Meshgrid size                                | `192`                      |
| `--latent_size`         | Size of latent vector                        | `128`                      |
| `--batch_size_reconstruct` | Batch size for reconstruction             | `100000`                   |


Or complete a partial mesh by inferring the input latent vectors:
``` bash
python complete_shape.py --device cuda
```

| Argument                | Description                                  | Default Value              |
|-------------------------|----------------------------------------------|----------------------------|
| `--weights_dir`         | Model weights directory                      | `out/weights`              |
| `--seed`                | Random seed                                  | `42`                       |
| `--device`              | Device to run on                             | `cpu`                      |
| `--batch_size`           | Batch size for training                              | `5`       |
| `--num-workers`         | Number of workers                            | `4`                        |
| `--lr_model`         | Learning rate to optimize the model           | `0.0001`                        |
| `--lr_latent`         | Learning rate to optimize the latent vector     | `0.001`                        |
| `--epochs`         | Number of epochs                            | `2000`                        |
| `--delta`         | Delta for clamping loss function        | `0.1`                        |
| `--latent_size`         | Size of latent vector                        | `128`                      |
| `--latent_std`         | Latent standard deviation                | `0.01`                      |
| `--nr_rand_samples`           | Number of random subsamples from each batch    | `10000`       |

And visualize the results with `visualize.py`. Modifications inside the source file may be necessary. Additionally, code for generating plots and evaluations can be found in `eval.py`.


## Output

The output of the processing pipeline is organized into several folders within the `out` directory.

- `1_preprocessed`: Contains `.npz` files with samples and signed distance functions (SDFs) generated from `.obj` mesh files as well as the normalized and centered `.obj` meshes. These are the raw data used for further processing and training.

- `2_reconstructed`: Holds the mesh files that have been reconstructed from the SDF samples. 

Additional directories are:

- `losses`: Contains the training plots of each training section.

- `screenshots`: Screenshots taken with the Visualizer after running `visualize.py`.

- `shape_completion`: All files generated for inferring shapes from partial pointclouds. Outputs from PointCleanNet are also saved here.

## Contributing
The Authors are [Duc Thinh Nguyen](https://github.com/duc-ng) (UCL) and [Harshitha Nagarajan ](https://github.com/HarshithaNagarajan) (UCL).

We would like to thank the following projects and their contributors for making their resources available for public use, which greatly facilitated the development of our project:

- **[DeepSDF](https://github.com/facebookresearch/DeepSDF)**: Original implementation of DeepSDF.
- **[PointCleanNet](https://github.com/mrakotosaon/pointcleannet)**: Source code integrated into our project.
- **[Objaverse](https://objaverse.allenai.org/)**: Provision of the training data.