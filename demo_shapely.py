# -*- coding: utf-8 -*-
"""
Demo of polygons handling and plotting using shapely & descartes.

"""

from WoodProblemDefinition import Stock, Order1, Order2, Order3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as pltcol

import shapely
from descartes import PolygonPatch


# Simple helper class for getting matplotlib patches from shapely polygons with different face colors
class PlotPatchHelper:
    # a colormap with 41 colors
    CMapColors = np.array([
        [0, 0.447, 0.741, 1],
        [0.85, 0.325, 0.098, 1],
        [0.929, 0.694, 0.125, 1],
        [0.494, 0.184, 0.556, 1],
        [0.466, 0.674, 0.188, 1],
        [0.301, 0.745, 0.933, 1],
        [0.635, 0.078, 0.184, 1],
        [0.333333333, 0.333333333, 0, 1],
        [0.333333333, 0.666666667, 0, 1],
        [0.666666667, 0.333333333, 0, 1],
        [0.666666667, 0.666666667, 0, 1],
        [1, 0.333333333, 0, 1],
        [1, 0.666666667, 0, 1],
        [0, 0.333333333, 0.5, 1],
        [0, 0.666666667, 0.5, 1],
        [0, 1, 0.5, 1],
        [0.333333333, 0, 0.5, 1],
        [0.333333333, 0.333333333, 0.5, 1],
        [0.333333333, 0.666666667, 0.5, 1],
        [0.333333333, 1, 0.5, 1],
        [0.666666667, 0, 0.5, 1],
        [0.666666667, 0.333333333, 0.5, 1],
        [0.666666667, 0.666666667, 0.5, 1],
        [1, 0, 0.5, 1],
        [1, 0.333333333, 0.5, 1],
        [1, 0.666666667, 0.5, 1],
        [1, 1, 0.5, 1],
        [0, 0.333333333, 1, 1],
        [0, 0.666666667, 1, 1],
        [0, 1, 1, 1],
        [0.333333333, 0, 1, 1],
        [0.333333333, 0.333333333, 1, 1],
        [0.333333333, 0.666666667, 1, 1],
        [0.333333333, 1, 1, 1],
        [0.666666667, 0, 1, 1],
        [0.666666667, 0.333333333, 1, 1],
        [0.666666667, 0.666666667, 1, 1],
        [0.666666667, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 0.333333333, 1, 1],
        [1, 0.666666667, 1, 1]
    ])

    # Alpha controls the opaqueness, Gamma how darker the edge line will be and LineWidth its weight

    def __init__(self, Gamma=1.3, Alpha=0.9, LineWidth=2.0):
        self.Counter = 0
        # darker edge color if Gamma>1 -> faceColor ** Gamma; use np.inf for black
        self.Gamma = Gamma
        self.Alpha = Alpha          # opaqueness level (1-transparency)
        self.LineWidth = LineWidth  # edge weight

    # circles through the colormap and returns the FaceColor and the EdgeColor (as FaceColor^Gamma)
    def nextcolor(self):
        col = self.CMapColors[self.Counter, :].copy()
        self.Counter = (self.Counter+1) % self.CMapColors.shape[0]
        return (col, col**self.Gamma)

    # returns a list of matplotlib.patches.PathPatch from the provided shapely polygons, using descartes; a list is
    # returned even for a single polygon for common handling
    def get_patches(self, poly):
        # single polygon, make it a one element list for common handling
        if not isinstance(poly, list):
            poly = [poly]
        patchList = []
        for p in poly:
            fCol, eCol = self.nextcolor()
            patchList.append(PolygonPatch(p, alpha=self.Alpha, facecolor=fCol, edgecolor=eCol,
                                          linewidth=self.LineWidth))
        return patchList


# Plots one or more shapely polygons in the provided axes ax. The named parameter values **kwargs are passed into
# PlotPatchHelper's constructor, e.g. you can write plotshapelyPoly(ax, poly, LineWidth=3, Alpha=1.0). Returns a list
# with the drawn patches objects even for a single polygon, for common handling
def plotshapelyPoly(ax, poly, **kwargs):
    return [ax.add_patch(p) for p in PlotPatchHelper(**kwargs).get_patches(poly)]


if __name__ == "__main__":
    # in case someone tries to run it from the command-line...
    plt.ion()

    # View the first stock together with pieces of the first order:
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Original Stock[0] and Order1')
    plotshapelyPoly(ax, Stock[0:1]+Order1)
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal')
    # In the above, ax.relim & ax.autoscale_view are both needed to rescale the limits to show all drawn patches
    # ax.set_aspect('equal') makes the X & Y axes to have the same data plot scaling (by default, each axis is
    # plotted with different scaling, so as the axes fill the figure)

    # In the previous example, the pieces were plotted one on top of other, because their starting coordinates match.
    # We can shift the various pieces to the right, in order to see them side-by-side including some extra space.
    # Considering Order3 as an example:
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Shifted Order3 pieces for better viewing')
    shifted = Order3.copy()
    for i in range(1, len(shifted)):
        # previous Xmax of bounding box (bounds property) plus 0.5 space
        xshift = shifted[i-1].bounds[2] + 0.5
        shifted[i] = shapely.affinity.translate(shifted[i], xshift)
    plotshapelyPoly(ax, shifted)
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal')

    # Lets consider the 7th stock and the 2nd order. We consider here random shifts and rotation of the order pieces
    # and we cut them of (== set difference) from the stock:
    np.random.seed(13)
    newOrder = [shapely.affinity.rotate(
        shapely.affinity.translate(
            Order3[i], xoff=6*np.random.rand(), yoff=6*np.random.rand()),
        360*np.random.rand(), origin='centroid') for i in range(len(Order3))]
    remaining = Stock[6]
    for p in newOrder:
        remaining = remaining.difference(p)
    # NOTE: the above operation is perhaps faster if we perform a cascade union first as below, check it on your code:
    #remaining = Stock[6].difference(shapely.ops.cascaded_union(newOrder))
    fig, ax = plt.subplots(ncols=2)
    fig.canvas.set_window_title(
        'Stock[6] cutting Order3 (translated & rotated)')
    pp = plotshapelyPoly(ax[0], Stock[6:7]+newOrder)
    pp[0].set_facecolor([1, 1, 1, 1])
    plotshapelyPoly(ax[1], remaining)
    ax[0].set_title('Rotated & translated order')
    ax[1].set_title('Remaining after set difference')
    ax[0].relim()
    ax[0].autoscale_view()
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    for a in ax:
        a.set_aspect('equal')
    # Initial stock was a Polygon, new one is a MultiPolygon (if not, try other random seeds):
    print(f"Original Stock[6] type: {type(Stock[6])}")
    print(f"Remaining piece type: {type(remaining)}")
    # We can split easily it into a list of simple polygons if needed, using list(<Multipolygon>):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Simple polygons from multi-polygon')
    plotshapelyPoly(ax, list(remaining))
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal')

    # Lets erode (shrink, negative buffer) or dilate (expand, positive buffer) the cut stock. See shapely manual for
    # the cap/join style options:
    eroded = remaining.buffer(-0.3,
                              join_style=shapely.geometry.JOIN_STYLE.mitre)
    dilated = remaining.buffer(
        0.3, join_style=shapely.geometry.JOIN_STYLE.mitre)
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Polygons buffering')
    # last in list gets plotted on top
    plotshapelyPoly(ax, [dilated, remaining, eroded], Alpha=0.6)
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal')

    # Finally, lets erode and then dilate the remaining with the same distance, known as opening in mathematical
    # morphology. Compare it with the initial remaining multi-polygon:
    joinStyle = shapely.geometry.JOIN_STYLE.mitre
    opening = remaining.buffer(-0.3,
                               join_style=joinStyle).buffer(0.3, join_style=joinStyle)
    fig, ax = plt.subplots(ncols=2)
    fig.canvas.set_window_title('Morphological opening')
    plotshapelyPoly(ax[0], remaining)
    plotshapelyPoly(ax[1], opening)
    ax[0].set_title('Original polygon')
    ax[1].set_title('Morphological opening')
    for a in ax:
        a.relim()
        a.autoscale_view()
        a.set_aspect('equal')
