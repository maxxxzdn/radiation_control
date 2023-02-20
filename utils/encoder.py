import numpy as np
from par_utils import *


def parabola_lambda(h, k, a):
    """
    Returns a function that computes parabola y = a(x-h)^2 + k .
    """
    return lambda x: a*(x-h)**2 + k

def round_to_int(x):
    """
    Round an array of floats to the nearest integer.
    """
    return np.round(x).astype(int)

def encode(image, int_distance, num_pars, threshold):
    """
    Encodes an image into a set of parabolas with a given distance between their centers along y-axis.
    The parabolas are encoded as a set of parameters (h,k,a) and a set of values for each pixel (x,y) of the parabola.
    Input:
        image (np.array): image to be encoded
        int_distance (int): distance between the centers of the parabolas along y-axis
        num_pars (int): number of parabolas
        threshold (float): intensity threshold for the intersection of the parabola and y-axis
    Output:
        (np.array): parameters of the parabolas
        (np.array): values of the parabolas
    """
    # compute parameters of each parabola
    par_params = get_n_parabolas(image, int_distance, num_pars, threshold)
    # functional forms of parabolas
    par_funcs = [parabola_lambda(*parabola) for parabola in par_params]

    # let us find pixels that belong to each parabola
    x = np.arange(0, 101,1)
    ys = []
    for par_func in par_funcs:
        y = par_func(x)
        # we only want to keep the positive part of the parabola (the part that is above the x-axis)
        y[y<0] = 0
        # round to integer
        y = round_to_int(y)
        ys.append(y)

    # we want to find the pixels that belong to each parabola
    intensity = []
    for y in ys:
        intensity.append(image.T[y, x])

    return np.concatenate(par_params).reshape(-1,3), np.concatenate(intensity).reshape(-1,101)

def decode(par_params, intensity):
    """
    Decodes a set of parabolas into an image.
    Input:
        par_params (np.array): parameters of the parabolas
        intensity (np.array): values of the parabolas
    Output:
        (np.array): decoded image
    """
    # functional forms of parabolas
    par_funcs = [parabola_lambda(*parabola) for parabola in par_params]

    # let us find pixels that belong to each parabola
    x = np.arange(0, 101,1)
    ys = []
    coords = []
    for par_func in par_funcs:
        y = par_func(x)
        # we only want to keep the positive part of the parabola (the part that is above the x-axis)
        y[y<0] = 0
        # round to integer
        y = round_to_int(y)
        ys.append(y)
        coords.append(np.array([x, round_to_int(y)]).T)

    # we want to find the pixels that belong to each parabola
    # each pixel is assigned the corresponding intensity
    image = np.zeros((101, 200))
    for val, y in zip(intensity, ys):
        image[x, y] = val     

    # we want to interpolate the image to fill up the gaps between the parabolas
    for loc in x:
        cs = [coord[loc] for coord in coords]
        x_int, f_int = interpolate_all([c[1] for c in cs], [image.item(*c) for c in cs])
        x_int = np.array(x_int)
        f_int = np.array(f_int)
        # only leave those elements for which x_int > 0 (the part that is above the x-axis)
        f_int = f_int[x_int > 0]
        x_int = x_int[x_int > 0]
        image[loc, x_int] = f_int

    return image