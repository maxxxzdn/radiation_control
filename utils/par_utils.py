import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def gauss_filter(x, sigma=1):
    """
    Gaussian filter applied to 1D data.
    """
    return gaussian_filter1d(x, sigma)

def insert_points(x1, x2, n):
    """
    Takes two points and adds n equidistant points between them (one dimensional integers).
    Input:
        x1 (int): start point
        x2 (int): end point
        n (int): number of points to be added in between
    Output:
        (list): list of points
    """
    n += 1
    return [x1 + i*(x2-x1)//n for i in range(n+1)]

def interpolate_all(x, y):
    """
    Interpolates between all the points in a list of points.
    Input:
        x (list): list of integer coordinates of points
        y (list): list of values of points
    Output:
        (list): list of integer coordinates of all intermediate points
        (list): list of linearly interpolated values of all intermediate points
    """
    x_new = np.arange(x[0], x[-1]+1, 1)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new

def vertex_x(image):
    """
    Find the x-coordinate vertex of the parabola based on the image.
    Uses gaussian filter to smooth the image and then finds the peak.
    """
    return find_peaks(gauss_filter(image.mean(1)), distance=image.shape[0]//2)[0].item()

def intersection_y(image, x, y, threshold):
    """
    Find the y-coordinate intersection of the parabola and y-axis.
    Input:
        image (np.array): image
        x (int): x-coordinate of the vertex of the parabola
        y (int): y-coordinate of a line that is perpendicular to x-axis and passes through the parabola
        threshold (float): intensity threshold for the intersection of the parabola and y-axis
    Output:
        (list): y-coordinates of the intersection of the parabola and y-axis
    """
    # intersection of the parabola and y-axis
    line = image[x:, y]
    # min-max normalization
    line = (line - line.min())/(line.max() - line.min())
    # find the first and the last element along the intersection whose intensity is bigger than the threshold
    peaks = [find_nearest(line, threshold), len(line) - find_nearest(line[::-1], threshold)]
    return peaks

def intersection_x(image, x, threshold):
    """
    Find the x-coordinate intersection of the parabola and y-axis.
    Input:
        image (np.array): image
        x (int): x-coordinate of the vertex of the parabola
        threshold (float): intensity threshold for the intersection of the parabola and y-axis
    Output:
        (list): x-coordinates of the intersection of the parabola and x-axis
    """
    # intersection of the parabola and x-axis
    line = image[x, :]
    # min-max normalization
    line = (line - line.min())/(line.max() - line.min())
    # find the first and the last element along the intersection whose intensity is bigger than the threshold
    peaks = [find_nearest(line, threshold), len(line) - find_nearest(line[::-1], threshold)]
    return peaks

def find_nearest(array, value):
    """
    Find the index of the first element in an array that is bigger than a given value.
    """
    for i, x in enumerate(array):
        if x > value:
            return i
    return len(array)

def get_n_parabolas(image, int_distance, num_pars, threshold=0.05):
    """
    Find the parameters of n parabolas that are present in the image.
    First, the algorithm finds vertices of each parabola from intensity pattern.
    Second, it analyzes the intersection of the parabola and y = y_vertex - int_distance 
        to discover the second point for each parabola from which the equation is recovered.
    Input:
        image (np.array): image
        int_distance (int): distance between the center of a parabola and 
        num_pars (int): number of parabolas
        threshold (float): intensity threshold for the intersection of the parabola and y-axis
    Output:
        (list): list of tuples containing the parameters of the parabolas
    """
    # first we find the axis of mirror symemtry of the parabola
    h = vertex_x(image)
    # along this axis we find a point with the highest intensity (the vertex of the main parabola)
    #   and multiple pixels for outlining parabolas
    k = [image[h].argmax()+i for i in range(-num_pars//2+1, num_pars//2+1)]
    # find the intersections of the parabola and y = y_vertex - int_distance
    peaks = intersection_y(image, h, k[num_pars//2]-int_distance, threshold)
    # insert n equidistant points between each pair of intersections
    peaks = insert_points(*peaks, num_pars-2)
    peaks = [i+h for i in peaks]

    # infer the "a" parameter for each parabola
    a = [(-int_distance)/(p-h)**2 for p in peaks]

    params = [(h,k,a) for k,a in zip(k,a)]  
    return params