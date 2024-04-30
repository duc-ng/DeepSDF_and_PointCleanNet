import numpy as np
import open3d as o3d
import os
from glob import glob


class Visualizer:
    """
    Class to display a meshes or pcds.
    """

    def __init__(self):
        self.o3d_meshes = []
        self.out_dir = "out/screenshots"
        os.makedirs(self.out_dir, exist_ok=True)

    def on_init(self, vis):
        vis.show_axes = False

    def take_screenshot(self, vis):
        out_file = os.path.join(self.out_dir, f"{len(os.listdir(self.out_dir))}.png")
        vis.export_current_image(out_file)

    def add_pcds(self, vertices=[], names=[]):
        for i, mesh in enumerate(vertices):
            mesh_o3d = o3d.geometry.PointCloud()
            mesh_o3d.points = o3d.utility.Vector3dVector(mesh)
            mesh_o3d.paint_uniform_color(np.random.rand(3))
            self.o3d_meshes.append(
                {"name": f"{names[i]}", "geometry": mesh_o3d, "is_visible": False}
            )

    def view(self):
        o3d.visualization.draw(
            self.o3d_meshes,
            show_ui=True,
            show_skybox=False,
            actions=[("Screenshot", lambda vis: self.take_screenshot(vis))],
            on_init=lambda vis: self.on_init(vis),
        )

    def add_sdf(self, npz_file):
        samples_and_sdfs = np.load(npz_file)
        samples = samples_and_sdfs["samples"]
        sdf = samples_and_sdfs["sdf"]
        negatives = samples[sdf < 0]
        positives = samples[sdf > 0]
        boundary = samples[sdf == 0]
        self.add_pcds(
            [positives, negatives, boundary],
            ["SDF: Outside", "SDF: Inside", "SDF: Boundary"],
        )

    def add_obj(self, mesh_files, with_time=False):
        color = np.random.rand(3)
        for i, mesh_file in enumerate(mesh_files):
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            mesh.paint_uniform_color(color)
            if i==0:
                mesh.paint_uniform_color(np.random.rand(3))
            mesh.compute_vertex_normals()
            print(
                f"Mesh {i}: {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces"
            )
            if with_time:
                self.o3d_meshes.append(
                    {"name": f"mesh {i}", "geometry": mesh, "time": i, "is_visible": False}
                )
            else:
                self.o3d_meshes.append({"name": f"mesh {i}", "geometry": mesh, "is_visible": False})


if __name__ == "__main__":
    # init visualizer
    np.random.seed(43)
    vis = Visualizer()
    shape = "bunny"

    # add SDF
    vis.add_sdf(f"out/1_preprocessed/{shape}.npz")

    # add reconstructions
    mesh_files = sorted(glob(f"out/2_reconstructed/{shape}/*.obj"))
    checkpoints = [int(os.path.basename(f).split(".")[0]) for f in mesh_files]
    mesh_files = [f"out/1_preprocessed/{shape}.obj"] + [f for _, f in sorted(zip(checkpoints, mesh_files))]
    vis.add_obj(mesh_files, with_time=False)

    # add more
    # vis.add_obj(["out/shape_completion/shape_completion.obj"], with_time=False)

    # show 3D window
    vis.view()
