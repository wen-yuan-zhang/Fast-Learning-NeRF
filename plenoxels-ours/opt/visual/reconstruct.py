import mcubes
import trimesh
import numpy as np
import torch
import os


def extract_mesh(links, density_data, filename='mesh.ply', filedir='visual'):
    """

    :param links: [M]
    :param density_data: [M, 1]
    :return:
    """

    reso = links.shape[0]

    # ???
    sigma = (links > 0).cpu().detach().numpy()

    sigma = density_data[:, 0][links.long()]
    sigma = sigma.cpu().detach().numpy()

    threshold = 25
    # print('fraction occupied', np.mean(sigma > threshold))
    vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    print('done', vertices.shape, triangles.shape)


    mesh = trimesh.Trimesh(vertices, triangles)
    mesh_filename = os.path.join(filedir, filename)
    mesh.export(mesh_filename)