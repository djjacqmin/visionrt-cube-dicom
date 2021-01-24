# visionrt-cube-dicom.py
import numpy as np

# Parameters for Cube Geometry
cube_half_width = 75  # mm
cube_inside_bevel = 67  # mm
bevel_radius = cube_half_width - cube_inside_bevel
cube_HU = -15

sphere_diameter = 7.5  # mm
sphere_centers = np.array(
    [[-10, -25, -30], [10, -15, -15], [0, 0, 0], [-20, 20, 10], [20, 30, 20]]
)
sphere_HU = 3060

# Parameters to jitter cube surface to make it imperfect
include_jitter = False
max_jitter = 0.02  # mm

# Parameters for Image Geometry
image_matrix_size = 256
image_physical_size = 200  # mm
image_pixel_size = image_physical_size / image_matrix_size
image_positions = (
    np.arange(image_matrix_size) * image_pixel_size
    + image_pixel_size / 2
    - image_physical_size / 2
)


# Parameters for Slice Geometry
number_of_slices = 256
image_physical_length = 200  # mm
slice_width = image_physical_length / number_of_slices  # plus one?
slice_positions = (
    np.arange(number_of_slices) * slice_width
    + slice_width / 2
    - image_physical_length / 2
)


def is_in_sphere(loc):
    """Checks whether the point loc is in the cube

    PARAMETERS
    ----------
    loc : list of float
        A list of length 3 corresponding to the x, y and z coordinates of the point

    RETURNS
    -------
    True if the point is in the a sphere, else False

    """

    assert len(loc) == 3, "loc must be a list of three floating point integers"

    for origin in sphere_centers:
        vec = loc - origin
        dist = np.linalg.norm(vec)
        if dist <= sphere_diameter / 2:
            return True

    return False


def is_in_cube(loc, rounded_edges=True):
    """Checks whether the point loc is in the cube

    PARAMETERS
    ----------
    loc : list of float
        A list of length 3 corresponding to the x, y and z coordinates of the point
    rounded_edges: bool
        If True, evaluates for a cube with rounded edges. If false, assumes cube
        has sharp edges (default is True)

    RETURNS
    -------
    True if the point is in the cube, else False

    """

    assert len(loc) == 3, "loc must be a list of three floating point integers"

    # Cube is symmetric, take abs of x,y,z, convert to ndarray which allows slicing
    loc = np.abs(loc)

    # Case 1: Any of the values are greater than the cube half-width, return false:
    if any(loc > cube_half_width):
        return False

    # If we are not using rounded edges, then we are done. Return True
    if not rounded_edges:
        return True

    # Case 2: All of the values are less than the cube_inside_bevel, return True:
    if all(loc <= cube_inside_bevel):
        return True

    # Case 3: If exactly one dimension is greater than cube_inside_bevel,
    # then the point is adjacent to a cube face, return True:
    if sum(loc > cube_inside_bevel) == 1:
        return True

    # Case 4: If exactly two dimensions are greater than cube_inside_bevel,
    # then the point is adjacent to a cube edge. The point defined by the two
    # dimensions that are greater than the cube_inside_bevel must be within a
    # radius of bevel_radius of the point cube_inside_bevel*[1,1]:
    if sum(loc > cube_inside_bevel) == 2:

        # Get the indicies whose value is greater than cube_inside_bevel
        tf_array = loc > cube_inside_bevel

        # Create a 2D vector from the origin to the point in the edge region
        origin = cube_inside_bevel * np.ones((3, 1))

        vec = loc[tf_array] - origin[tf_array]
        dist = np.linalg.norm(vec)
        if dist <= bevel_radius:
            return True
        else:
            return False

    # Case 5: If exactly three dimensions are greater than cube_inside_bevel,
    # then the point is adjacent to a cube corner. The point must be within a
    # radius of bevel_radius of the point cube_inside_bevel*[1,1,1]:
    if sum(loc > cube_inside_bevel) == 3:
        origin = cube_inside_bevel * np.ones((3, 1))
        vec = loc - origin
        dist = np.linalg.norm(vec)
        if dist <= bevel_radius:
            return True
        else:
            return False


def is_in_sphere_np(x, y, z):
    """

    PARAMETERS
    ----------
    x : list of float
        A list of image coordinates in the x-dimension
    y : list of float
        A list of image coordinates in the y-dimension
    z : list of float
        A list of image coordinates in the z-dimension

    RETURNS
    -------
    A 3D array of shape (len(x),len(y),len(z)), where the array value (i,j,k) is True 
    if the point x[i], y[j], z[k] is locted in a in the a sphere, else False

    """

    XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")

    is_in_sphere_array = np.zeros(XX.shape, dtype=bool)

    for origin in sphere_centers:
        DXX = XX - origin[0]
        DYY = YY - origin[1]
        DZZ = ZZ - origin[2]

        quad_sum = np.square(DXX) + np.square(DYY) + np.square(DZZ)

        is_in_this_sphere = quad_sum <= (sphere_diameter / 2) ** 2

        np.logical_or(is_in_this_sphere, is_in_sphere_array, out=is_in_sphere_array)

    return is_in_sphere_array
