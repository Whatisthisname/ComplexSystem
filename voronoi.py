# Source - https://stackoverflow.com/a
# Posted by pv., modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-26, License - CC BY-SA 3.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_triples(pairs, values, colorbarlabel : str, xlabel : str, ylabel : str):
    vor = Voronoi(pairs)
    regions, vertices = voronoi_finite_polygons_2d(vor)


    # colorize
    for i, (region, frac) in enumerate(zip(regions, values)):
        polygon = vertices[region]
        c = plt.cm.viridis(frac)
        plt.fill(*zip(*polygon), alpha=1.0, color=c, linewidth=1)

    plt.plot(pairs[:,0], pairs[:,1], 'ko', markersize=2)
    plt.xlim(vor.min_bound[0], vor.max_bound[0])
    plt.ylim(vor.min_bound[1], vor.max_bound[1])
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap='viridis'), ax=plt.gca(), label=colorbarlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()

# Example usage
if __name__ == "__main__":

    # make up data points
    np.random.seed(1234)
    points = np.random.rand(15, 2)

    xs = np.linspace(0, 1, 11)
    ys = np.linspace(0, 1, 13)
    X, Y = np.meshgrid(xs, ys)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    fraction = (points[:, 0]** 2 + points[:, 1]**2)

    xs = np.linspace(0.3, 0.6, 19)
    ys = np.linspace(0.3, 0.6, 19)
    X, Y = np.meshgrid(xs, ys)
    points2 = np.vstack([X.ravel(), Y.ravel()]).T
    fraction2 = (points2[:, 0]** 2 + points2[:, 1]**2)


    points = np.vstack([points, points2])
    fraction = np.concatenate([fraction, fraction2])

    xs = np.linspace(0, 1, 11)
    ys = np.linspace(0, 1, 13)


    plot_triples(pairs = points, values=fraction, colorbarlabel="Some Value", xlabel="X-axis", ylabel="Y-axis")

