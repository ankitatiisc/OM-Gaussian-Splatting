import numpy as np
from plyfile import PlyData, PlyElement
import numpy as np
import pyvista as pv
from pyvista import examples
# Assuming you have your point cloud data stored in numpy arrays
# For example, points would be an Nx3 array containing XYZ coordinates
# and colors would be an Nx3 array containing RGB values (0-255 range)
def save_as_colorply(points,color):
    # Create PlyElement for vertices
    vertex = PlyElement.describe(points, 'vertex')

    # Add color property to the vertex
    vertex.properties.append(('red', 'uchar'))
    vertex.properties.append(('green', 'uchar'))
    vertex.properties.append(('blue', 'uchar'))

    # Combine vertices and colors
    vertex.data['red'] = colors[:, 0]  # Assuming colors is Nx3 with RGB values
    vertex.data['green'] = colors[:, 1]
    vertex.data['blue'] = colors[:, 2]

    # Save PlyData
    ply_data = PlyData([vertex])
    ply_data.write('point_cloud_with_color.ply')
    


def save_as_ply(points, filename):
    # Ensure points is a numpy array
    points = np.array(points)
    import pdb;pdb.set_trace()
    # Create PlyElement for vertices
    vertex = PlyElement.describe(points, 'vertex')

    # Save PlyData
    ply_data = PlyData([vertex])
    ply_data.write(filename)

# Example usage
points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Example point cloud data

save_as_ply(points, 'point_cloud.ply')