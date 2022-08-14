import pydevd_pycharm
pydevd_pycharm.settrace('166.111.81.117', port=56453, stdoutToServer=True, stderrToServer=True)
import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import imageio
import pprint

import matplotlib.pyplot as plt

import run_nerf
import run_nerf_helpers
from run_nerf import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = config_parser()
args = parser.parse_args()
render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

# Render an overhead view to check model was loaded correctly
c2w = torch.tensor(np.eye(4)[:3,:4].astype(np.float32)) # identity pose matrix
c2w[2,-1] = 4.
H, W, focal = 800, 800, 1200.
down = 8
H, W, focal = H // down, W // down, focal // down
K = np.array([[focal, 0, 0.5*W],
              [0, focal, 0.5*H],
              [0, 0, 1]])
with torch.no_grad():
    test = run_nerf.render(H, W, K, c2w=c2w, near=2., far=6., **render_kwargs_test)
img = np.clip(test[0].cpu().detach().numpy(),0,1)
plt.imshow(img)
plt.show()


# Query network on dense 3d grid of points

N = 256
t = torch.linspace(-1.2, 1.2, N + 1)

query_pts = torch.stack(torch.meshgrid(t, t, t), -1)
print(query_pts.shape)
sh = query_pts.shape
flat = query_pts.reshape([-1, 3])


# fn = lambda i0, i1: net_fn(flat[i0:i1, None, :], viewdirs=,
#                            network_fn=render_kwargs_test['network_fine'])
chunk = 1024 * 64
network_query_fn = render_kwargs_test['network_query_fn']
network_fine = render_kwargs_test['network_fine']
outputs = []
for i in range(0, flat.shape[0], chunk):
    with torch.no_grad():
        chunk_output = network_query_fn(flat[i: i+chunk, None, :], torch.zeros_like(flat[i: i+chunk]), network_fine).cpu().detach().numpy()
    outputs.append(chunk_output)
raw = np.concatenate(outputs, 0)
raw = np.reshape(raw, list(sh[:-1]) + [-1])
sigma = np.maximum(raw[..., -1], 0.)

print(raw.shape)
plt.hist(np.maximum(0, sigma.ravel()), log=True)
plt.show()


# Marching cubes with PyMCubes

import mcubes

threshold = 50.
print('fraction occupied', np.mean(sigma > threshold))
vertices, triangles = mcubes.marching_cubes(sigma, threshold)
print('done', vertices.shape, triangles.shape)

### Uncomment to save out the mesh
# mcubes.export_mesh(vertices, triangles, "logs/lego_example/lego_{}.dae".format(N), "lego")

import trimesh

mesh = trimesh.Trimesh(vertices / N - .5, triangles)
basedir = args.basedir
expname = args.expname
mesh_filename = 'lego_mesh.ply'
mesh.export(os.path.join(basedir, expname, mesh_filename))
mesh.show()