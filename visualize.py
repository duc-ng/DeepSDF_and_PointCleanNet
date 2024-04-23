import numpy as np
import open3d as o3d
import os


class Visualizer:
    """
    Class to display a meshes or pcds.
    """

    def __init__(self):
        self.o3d_meshes = []
        self.out_dir = "out/screenshots"
        os.makedirs(self.out_dir, exist_ok=True)
        self.colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ] 

    def on_init(self, vis):
        vis.show_axes = False

    def take_screenshot(self, vis):
        out_file = os.path.join(self.out_dir, f"{len(os.listdir(self.out_dir))}.png")
        vis.export_current_image(out_file)

    def add_pcds(self, vertices=[], names=[]):
        for i, mesh in enumerate(vertices):
            mesh_o3d = o3d.geometry.PointCloud()
            mesh_o3d.points = o3d.utility.Vector3dVector(mesh)
            mesh_o3d.paint_uniform_color(self.colors[i])
            self.o3d_meshes.append({"name": f"{names[i]}", "geometry": mesh_o3d})

    def view(self):
        o3d.visualization.draw(
            self.o3d_meshes,
            show_ui=True,
            show_skybox=False,
            actions=[("Screenshot", lambda vis: self.take_screenshot(vis))],
            on_init=lambda vis: self.on_init(vis),
        )

    def show_sdf(self, npz_file):
        samples_and_sdfs = np.load(npz_file)
        negatives = samples_and_sdfs["neg"][:, :3]
        positives = samples_and_sdfs["pos"][:, :3]
        self.add_pcds([positives, negatives], ["positives", "negatives"])

    def show_obj(self, mesh_files):
        for i, mesh_file in enumerate(mesh_files):
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            mesh.paint_uniform_color(self.colors[i])
            mesh.compute_vertex_normals()
            print(f"Mesh {i}: {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
            self.o3d_meshes.append({"name": f"mesh {i}", "geometry": mesh})


if __name__ == "__main__":
    # init visualizer
    vis = Visualizer()  

    # show SDF
    # vis.show_sdf("out/1_preprocessed/bunny.npz")

    # show reconstructions
    vis.show_obj(
        [
            "out/1_preprocessed/bunny.obj",
            "out/2_reconstructed/bunny.obj",
            "out/3_cleaned_outliers/bunny.obj",
            "out/4_cleaned_noise/bunny_1.obj",
            # "out/4_cleaned_noise/bunny_2.obj",
        ]
    )
    
    vis.view()
