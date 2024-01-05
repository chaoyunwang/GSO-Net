"""use to convert .obj to .npy or .npy to .obj"""
import pywavefront
import numpy as np
from data_process import save_mesh,quads

def obj2npy(input_obj, output_npy):
    obj = pywavefront.Wavefront(input_obj)
    vertices = obj.vertices
    points = np.reshape(np.array(vertices), (64, 64, 3))
    np.save(output_npy, points)

def npy2obj(input_npy, output_obj):
    points = np.load(input_npy)
    points=points.reshape(-1,3)
    quad = quads()
    save_mesh(output_obj, points, quad)

if __name__ == '__main__':
    #obj2npy
    obj_file = "./convert_example/Developable.obj"
    obj2npy(obj_file, obj_file.replace(".obj", ".npy"))

    #npy2obj
    npy_file = "./convert_example/Developable.npy"
    npy2obj(npy_file, npy_file.replace(".npy", ".obj"))

