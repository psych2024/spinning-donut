import os
import time
from math import pi, sin, cos, ceil, sqrt

import matplotlib.pyplot as plt

# ============ PART 1 ============= #
def create_vector(components: int, fill: float) -> list:
    """
    Create a new vector based on our specifications
    :param components: number of vector components
    :param fill: default number to set the each component to
    :return: A new vector
    """
    v = []
    for i in range(components):
        v.append([fill])
    return v

def test_create_vector():
    v = create_vector(3,1)

    # check dimensions correct
    for u in v:
        assert len(u) == 1
        assert u[0] == 1
    assert len(v) == 3

    # check deep copy
    t = create_vector(3, 1)
    assert v is not t

test_create_vector()

def vector_add(a: list, b:list) -> list:
    """
    Computes and returns a + b
    a and b should remain unchanged after the additon
    :param a: vector a
    :param b: vector b
    :return: a new vector object corresponding to a + b
    """
    result = create_vector(len(a), 0)
    for i in range(len(result)):
        result[i][0] = a[i][0] + b[i][0]
    return result

def test_add_vector():
    a = create_vector(3, 1)
    b = create_vector(3, -2)

    result = vector_add(a, b)
    assert (len(result) == len(a))
    for i in range(len(result)):
        assert result[i][0] == -1
        assert a[i][0] == 1
        assert b[i][0] == -2

test_add_vector()

def vector_scalar_multiplication(V: list, c: float) -> list:
    """
    Computes an returns c * V
    V should remain unchanged after the scalar multiplcation
    :param V: the vector
    :param c: the scalar
    :return: a new vector corresponding to c * V
    """
    result = create_vector(len(V), 0)
    for i in range(len(V)):
        result[i][0] = c * V[i][0]
    return result

def test_vector_scale():
    V = create_vector(3, 2.5)
    c = 2

    result = vector_scalar_multiplication(V, c)
    assert (len(result) == len(V))
    for i in range(len(result)):
        assert result[i][0] == 5
        assert V[i][0] == 2.5

test_vector_scale()

def dot_product(A: list, B: list) -> float:
    """
    Computes dot product of two vectors
    Raises ValueError if dimension is zero or not a vector (i.e. more than one column)
    :param A: Vector A
    :param B: Vector B
    :return: Dot product of A and B
    """

    if len(A) == 0 or len(B) == 0:
        raise ValueError("Vector dimension is zero")

    if len(A[0]) != 1 or len(B[0]) != 1:
        raise ValueError("Not a vector")

    if len(A) != len(B):
        raise ValueError("Dimensions don't match")

    R = 0
    for i in range(len(A)):
        R += A[i][0] * B[i][0]
    return R

def test_dot_product():
    a = [[1],[2],[3]]
    b = [[4],[5],[6]]
    prod = dot_product(a, b)
    assert prod == 1 * 4 + 2 * 5 + 3 * 6

    assert dot_product(create_vector(3, 0), create_vector(3, 1)) == 0

def get_magnitude(V: list) -> float:
    """
    Computes magnitude of vector
    Raises ValueError if dimension is zero or not a vector (i.e. more than one column)
    :param V: the vector
    :return: magnitude
    """
    if len(V) == 0:
        raise ValueError("Vector dimension is zero")

    if len(V[0]) != 1:
        raise ValueError("Not a vector")

    # alternatively: use dot product
    result = 0
    for row in V:
        result += row[0] ** 2
    return sqrt(result)

def test_get_magnitude():
    assert get_magnitude(create_vector(3, 0)) == 0
    assert get_magnitude(create_vector(3, 2)) == sqrt(12)
    assert get_magnitude([[1],[2],[3]]) == sqrt(1 + 4 + 9)
    b = [[4], [5], [6]]
    assert get_magnitude(b) == sqrt(dot_product(b, b))

def normalize(V: list) -> list:
    """
    Returns unit vector
    Raises ValueError if dimension is zero or not a vector (i.e. more than one column)
    Also raises ValueError if V is zero vector
    V should not be modified while normalizing
    :param V: the vector
    :return: unit vector in direction of V
    """
    if len(V) == 0:
        raise ValueError("Vector dimension is zero")

    if len(V[0]) != 1:
        raise ValueError("Not a vector")

    mag = get_magnitude(V)
    if mag == 0:
        raise ValueError("Cannot normalize zero vector")

    result = create_vector(len(V), 0)
    for i in range(len(V)):
        result[i][0] = V[i][0] / mag
    return result

EPS = 1e-6
def test_normalize():
    a = [[1], [2], [3]]
    b = [[2], [4], [6]]
    an = normalize(a)
    bn = normalize(b)
    assert abs(get_magnitude(an) - 1) < EPS
    assert abs(get_magnitude(bn) - 1) < EPS

    for i in range(1, 4):
        assert a[i - 1][0] == i
        assert abs(an[i - 1][0] - i/get_magnitude(a)) < EPS
        assert abs(bn[i - 1][0] - 2*i/get_magnitude(b)) < EPS
        assert abs(an[i - 1][0] - bn[i - 1][0]) < EPS

test_normalize()

print("Part 1 Complete!")

# ============ PART 2 ============= #
def n_matrix(row: int, col: int, n: float) -> list:
    """
    Creates and returns a [row x col] matrix with all entries set to n
    :param n: the value to fill in
    :param row: number of rows
    :param col: number of columns
    :return: new matrix object
    """
    return [[n for _ in range(col)] for _ in range(row)]

def n_matrix_test():
    m = n_matrix(2, 3, -1)
    assert len(m) == 2
    assert len(m[0]) == len(m[1]) == 3
    for row in m:
        for x in row:
            assert x == -1
    assert n_matrix(3, 1, 0) == create_vector(3, 0)

n_matrix_test()

def x_axis_rotation_matrix(theta: float) -> list:
    """
    Returns a matrix corresponding to the linear transformation that rotates
    entire 3D space about the x-axis
    :param theta: the angle
    :return: a matrix object
    """
    return [[1,               0,             0],
            [0,      cos(theta),   -sin(theta)],
            [0,      sin(theta),    cos(theta)]]

def y_axis_rotation_matrix(theta: float) -> list:
    """
    Returns a matrix corresponding to the linear transformation that rotates
    entire 3D space about the y-axis
    :param theta: the angle
    :return: a matrix object
    """
    return [[cos(theta),     0,            sin(theta)],
            [0,              1,                     0],
            [-sin(theta),    0,           cos(theta)]]
def z_axis_rotation_matrix(theta: float) -> list:
    """
    Returns a matrix corresponding to the linear transformation that rotates
    entire 3D space about the z-axis
    :param theta: the angle
    :return: a matrix object
    """
    return [[cos(theta), -sin(theta),           0],
            [sin(theta),  cos(theta),           0],
            [0,                    0,           1]]

def matrix_addition(A: list, B: list) -> list:
    """
    Adds two matrices
    Raises ValueError if any dimension is zero or if dimensions don't match
    :param A: matrix A as a list
    :param B: matrix B as a list
    :return: New matrix of A + B
    """
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Matrix dimension is zero")

    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Dimensions don't match")

    R = n_matrix(len(A), len(A[0]), 0)
    for i in range(len(A)):
        for j in range(len(A[0])):
            R[i][j] = A[i][j] + B[i][j]
    return R

def test_matrix_addition():
    a = n_matrix(3, 4, 2)
    b = n_matrix(3, 4, -2)
    c = matrix_addition(a, b)
    assert (len(c) == 3)
    for row in c:
        assert len(row) == 4
        for x in row:
            assert x == 0

    for row in a:
        for x in row:
            assert x == 2

    for row in b:
        for x in row:
            assert x == -2

test_matrix_addition()

def matrix_multiply(A: list, B: list) -> list:
    """
    Multiplies two matrices
    Raises ValueError if any dimension is zero or if dimensions don't match
    :param A: matrix A as a list
    :param B: matrix B as a list
    :return: New matrix of A * B
    """
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Matrix dimension is zero")

    if len(A[0]) != len(B):
        raise ValueError("Dimensions don't match")

    R = n_matrix(len(A), len(B[0]), 0)
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                R[i][j] += A[i][k] * B[k][j]
    return R

def test_matrix_multiply():
    zero = n_matrix(3, 3, 0)
    a = n_matrix(3,3, 2)
    assert matrix_multiply(a, zero) == zero

    a = [[1,2,3], [4,5,6], [7,8,9]]
    r = matrix_multiply(a,a)
    assert r[0][0] == 30
    assert r[0][1] == 36
    assert r[0][2] == 42
    assert r[1][0] == 66
    assert r[1][1] == 81
    assert r[1][2] == 96
    assert r[2][0] == 102
    assert r[2][1] == 126
    assert r[2][2] == 150

test_matrix_multiply()

print("Part 2 Complete!")

# ============ PART 3 ============= #
# system properties
SCREEN_WIDTH = 80
SCREEN_HEIGHT = 24
FRAMES_PER_SECOND = 10
ANIMATION_DELAY_SECONDS = 1 / FRAMES_PER_SECOND

# assume a screen at y = SCREEN_Y_VALUE
# default SCREEN_Y_VALUE = 18
SCREEN_Y_VALUE = 18

# animation properties
# assume viewer is at (0,0,0)
LIGHT_DIRECTION = normalize([[0],
                             [-1],
                             [1]])

STAGE_DISTANCE_UNITS = 5.0

# radius of the tube
TORUS_MINOR_RADIUS = 1.0
# distance from the center of the tube to the center of the torus
TORUS_MAJOR_RADIUS = 2.0

# default THETA_STEP = 0.07
THETA_STEP = 0.07

# default PHI_STEP = 0.03
PHI_STEP = 0.03


ILLUMINATION_CHAR = ".,-~:;=!*#$@"
MAX_ILLUMINATION = len(ILLUMINATION_CHAR)

def plot_3d_vectors(vectors: list) -> None:
    """
    Takes in a list of 3D vectors and plots it on a 3D graph using matplotlib
    :param vectors: The list of vectors
    """
    x = []
    y = []
    z = []
    for vec in vectors:
        x.append(vec[0][0])
        y.append(vec[1][0])
        z.append(vec[2][0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)
    plt.show()

def get_torus_points(alpha: float, beta: float) -> (list, list):
    """
    Calculates and returns
     1. all points on the donut (as vectors)
     2. The normal vectors of each point
    :param alpha: the angle for the first axis of rotation
    :param beta: the angle for the second axis of rotation
    :return: A tuple of length 2 storing two lists. The first list contains all points on the donut.
    The second list contains the normal vectors for each point
    """

    # list that stores the points on the donut
    points = []

    # list that stores the normal vectors for each point
    # Ex: the normal vector for the first point in the list is stored at norms[0]
    norms = []

    animation_rotations = matrix_multiply(x_axis_rotation_matrix(beta), y_axis_rotation_matrix(alpha))
    theta = 0
    while theta < 2 * pi:
        circle = [[TORUS_MINOR_RADIUS],
                  [0],
                  [0]]

        circle = matrix_multiply(y_axis_rotation_matrix(theta), circle)
        circle_norm = circle.copy()

        translation = [[TORUS_MAJOR_RADIUS],
                       [0],
                       [0]]
        circle = matrix_addition(circle, translation)
        theta += THETA_STEP

        phi = 0
        while phi < 2 * pi:
            all_rotations = matrix_multiply(animation_rotations, z_axis_rotation_matrix(phi))
            phi += PHI_STEP

            point = matrix_multiply(all_rotations, circle)
            normal = matrix_multiply(all_rotations, circle_norm)

            points.append(point)
            norms.append(normal)
    return points, norms

def get_luminance_index(percentage: float) -> int:
    """
    Gets the correct character to render for the point based on the luminance/brightness
    of the point
    :param percentage: the percentage of brightness
    :return: the index for the luminance character
    """
    step = 1.0 / float(MAX_ILLUMINATION)
    idx = ceil(percentage / step)
    idx = max(idx, 0)
    idx = min(idx, MAX_ILLUMINATION - 1)
    return idx

def render_donut(points: list, norms: list) -> list:
    """
    This function renders a specific screenshot of the donut on a screen
    :param points: all points on the donut
    :param norms: the normal vectors for each point
    :return: the snapshot of the donut given the points and normal vectors
    """
    screen = n_matrix(SCREEN_HEIGHT, SCREEN_WIDTH, -1)

    for i in range(len(points)):
        point = points[i]
        normal = norms[i]

        # add stage distance
        point[1][0] += STAGE_DISTANCE_UNITS

        constant = SCREEN_Y_VALUE / point[1][0]
        screen_x = int(point[0][0] * constant) + SCREEN_WIDTH // 2
        screen_z = int(point[2][0] * constant) + SCREEN_HEIGHT // 2

        dotp = dot_product(normalize(normal), LIGHT_DIRECTION)

        if dotp < 0:
            continue

        if not SCREEN_WIDTH > screen_x >= 0:
            continue

        if not SCREEN_HEIGHT > screen_z >= 0:
            continue

        screen[screen_z][screen_x] = max(screen[screen_z][screen_x], get_luminance_index(dotp))
    return screen

def redraw_screen(screen) -> None:
    """
    Prints a snapshot of the donut into the terminal
    :param screen: the snapshot of the donut
    """
    size = os.get_terminal_size()

    if SCREEN_WIDTH > size.columns or SCREEN_HEIGHT > size.lines:
        print(f"Terminal (width=${size.columns}, height=${size.lines}) is too small to render animation!")
        exit(0)

    rem = size.lines - SCREEN_HEIGHT

    for row in screen:
        for idx in row:
            if idx == -1:
                print(' ', end='')
            else:
                print(ILLUMINATION_CHAR[idx], end='')
        print('')

    for _ in range(rem):
        print('')

ALPHA_STEP = 0.07
BETA_STEP = 0.03
alpha = 0.2
beta = 1

print("precomputing")

stages = []
while alpha < 2 * pi or beta < 2 * pi:
    stages.append(render_donut(*get_torus_points(alpha, beta)))
    alpha += ALPHA_STEP
    beta += BETA_STEP

print("done!")

count = 0
while True:
    redraw_screen(stages[count % len(stages)])
    count += 1
    time.sleep(ANIMATION_DELAY_SECONDS)

