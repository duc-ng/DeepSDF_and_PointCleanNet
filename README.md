# DeepSDF_and_PointCleanNet

This project constitutes Coursework 3 within the curriculum of COMP0119 - Acquisition and Processing of 3D Geometry. We have developed the simple version of [DeepSDF](https://arxiv.org/abs/1901.05103) that is trained on single shapes and have further expanded upon this by integrating [PointCleanNet](https://geometry.cs.ucl.ac.uk/projects/2019/pointcleannet/) into our methodology for mesh reconstruction. Comprehensive details can be found in the submitted reports.

## Installation

Set up the project with the following commands:

``` bash
conda create -n deepsdf python=3.10 -y
conda activate deepsdf
pip install torch torchvision torchaudio open3d trimesh rtree scikit-image
```

Then download models:

``` bash
cd pointcleannet/models && python download_models.py --task denoising && python download_models.py --task outliers_removal && cd ../..
```

## Usage

Start with preprocessing your `.obj` data files by running:
``` bash
python preprocess .py
```
| Argument               | Description                                      | Default Value         |
|------------------------|--------------------------------------------------|-----------------------|
| `--input_dir`          | Directory containing the `.obj` files to process | `data`                |
| `--output_dir`         | Directory where the output `.npz` files will be saved | `out/1_preprocessed` |
| `--samples_on_surface` | Number of samples to take directly on the surface of the mesh | `15000`               |
| `--samples_in_bbox`    | Number of samples to take within the bounding box of the mesh | `15000`               |



You can train the SDF model with:
``` bash
python train.py
```
| Argument        | Description                                       | Default Value                     |
|-----------------|---------------------------------------------------|-----------------------------------|
| `--input_file`  | File to train on                                  | `out/1_preprocessed/bunny.npz`    |
| `--output_dir`  | Output directory                                  | `out/2_reconstructed`           |
| `--seed`        | Random seed                                       | `42`                              |
| `--device`      | Device to train on                                | `cpu`                             |
| `--batch-size`  | Input batch size for training                     | `12800`                           |
| `--num-workers` | Number of workers                                 | `8`                               |
| `--lr`          | Learning rate                                     | `0.0001`                           |
| `--epochs`      | Number of epochs to train                         | `100`                             |
| `--delta`       | Delta for clamping                                | `0.1`                             |
| `--N`           | Meshgrid size                                     | `128`                             |
| `--clean`       | Clean the reconstruction with PointCleanNet       | `false`                           |
| `--clean_nrun`  | Number of runs with PointCleanNet noise removal   | `2`                               |
| `--load`        | Load model weights from file instead of training  | `None`                            |


And visualize the results with `visualize.py`. Modifications inside the source file may be necessary.
``` bash
python visualize.py
```


## Output

The output of the processing pipeline is organized into several folders within the `out` directory.

- `1_preprocessed`: Contains `.npz` files with samples and signed distance functions (SDFs) generated from `.obj` mesh files as well as the normalized and centered `.obj` meshes. These are the raw data used for further processing and training.

- `2_reconstructed`: Holds the mesh files that have been reconstructed from the SDF samples. 

- `3_cleaned_outliers`: Contains processed mesh files where outliers have been removed with PointCleanNet.

- `4_cleaned_noise`: Includes the final, cleaned mesh files with noise reduction applied with PointCleanNet.

Each of these directories builds upon the previous step. We also have additional directories.

- `loss`: Contains the training plots of each training section.

- `screenshots`: Screenshots taken with the Visualizer after running `visualize.py`.

## Contributing
[Duc Thinh Nguyen](https://github.com/duc-ng)

[Harshitha Nagarajan ](https://github.com/HarshithaNagarajan)
