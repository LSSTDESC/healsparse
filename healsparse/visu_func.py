from __future__ import division, absolute_import, print_function
import numpy as np
import healpy as hp
from healsparse.healSparseMap import HealSparseMap
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from copy import copy
# Importing cartopy
try:
    import cartopy.crs
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
except ImportError:
    print('Cartopy is not available, cannot use cartopy projections')
    have_cartopy = False
# Importing shapely
try:
    import shapely.geometry as sgeom
except ImportError:
    print('Shapely not found, tick-labels on non-cylindrical projections are not available.')

def get_projection(projection_name, central_lon=0):
    """
    Auxiliary function to get the cartopy.crs projections from their name
    
    Parameters:
    -----------
    projection_name: `str`,
        Name of the cartopy.crs.Projection object to use
    central_lon: `float`,
        Central longitude of the projection (in degrees).
    """
    # Get the name of all cartopy's projections:
    if have_cartopy:
        valid_projections = {}
        for obj_name, o in vars(cartopy.crs).copy().items():
            if isinstance(o, type) and issubclass(o, cartopy.crs.Projection) and \
               not obj_name.startswith('_') and obj_name not in ['Projection']:
                valid_projections[obj_name]=o
        if projection_name in valid_projections:
            return getattr(cartopy.crs, projection_name)(central_lon)
        else:
            raise ValueError('The projection selected is not available. Try using one of these', valid_projections.keys())
    else:
        return None

# Routines to add labels on non-cylindrical projections
# These only work if shapely is installed and when the "extent" of the maps are not full sky
# Based on https://gist.github.com/ajdawson/dd536f786741e987ae4e (requires cartopy >=0.12)

def find_side(ls, side):
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.
    
    """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])

def custom_xticks(ax, ticks):
    """Draw ticks on the bottom x-axis of a cartopy projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _custom_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
    
def custom_yticks(ax, ticks):
    """Draw ricks on the left y-axis of a cartopy projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _custom_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])

def _custom_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of a cartopy projection."""
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(cartopy.crs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(cartopy.crs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:    
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels

# Below we define the routine to display the maps
        
def hsp_view_map(HealSparseMap, projection='Robinson', show_coverage=True, central_lon=0, extent=[], 
                 cmap_name='viridis', figsize=(10,5), savename=None, title=None, colorlabel=None,
                 vmin=None, vmax=None, nx=10, ny=10, **kwargs):
    """
    Method to visuzalize HealSparseMaps

    Parameters:
    -----------
    HealSparseMap: `HealSparseMap`,
        Input HealSparseMap to view.
    projection: `str`,
        Map projection to use for the visualization.
    show_coverage: `bool`,
        If `True` show the coverage mask.
    central_lon: `float`,
        Central longitude of the map (in degrees). Default: 0 deg.
    extent: `list`,
        List specifying the minimum RA and Dec to visualize (in degrees).
        Format: [ra_min, ra_max, dec_min, dec_max].
    cmap_name: `str`,
        Name of the matplotlib.colormap to use.
    figsize: `tuple`,
        Size of the figure to produce.
    savename: `str`,
        Path to save the figure.
    vmin: `float`,
        Minimum value for the color scale.
    vmax: `float`,
        Maximum value for the color scale.
    nx: `int`,
        Number of lines in longitude grid.
    ny: `int`,
        Number of lines in the latitude grid.
    **kwargs: `**kwargs`, arguments for `matplotlib.pyplot.subplots`
    or `matplotlib.pyplot.axis` objects
    """
    fig = plt.figure(figsize=figsize, frameon=True)
    proj = get_projection(projection, central_lon=central_lon)
    if show_coverage:
        ax = fig.add_subplot(121, projection=proj, **kwargs)
        ax_cov = fig.add_subplot(122)
    else:
        ax = fig.add_subplot(111, projection=proj, **kwargs)
    # If extent is specified but of incorrect length, raise an error
    if (len(extent)>1) & (len(extent)!=4):
        raise ValueError('If extent is specified it needs to have the format \
            ra_min, ra_max, dec_min, dec_max')
    elif len(extent)==4:
        xticks = np.linspace(extent[0], extent[1], nx).tolist()
        yticks = np.linspace(extent[2], extent[3], ny).tolist()
        # If extent is not specified, show the whole map
    else:
        extent = None
        xticks = np.linspace(-179.99999, 179.99999, nx).tolist()
        yticks = np.linspace(-89.99999, 89.99999, ny).tolist()
        # This shows the full globe
        ax.set_global()
    # Get the vectors that point to the position of the corners, reshape to feed into vec2ang to get the position
    # in ra, dec
    corner_ra, corner_dec = hp.vec2ang(np.transpose(hp.boundaries(HealSparseMap.nsideSparse, HealSparseMap.validPixels,
                                                                  nest=True), axes=[0,2,1]).reshape(-1,3), lonlat=True)
    # Now reshape to create the corresponding matplotlib polygons:
    uv_verts = np.array([corner_ra.reshape(-1, 4),
                     corner_dec.reshape(-1, 4)]).transpose(1, 2, 0)
    
    # Get the colormap
    cmap = plt.get_cmap(cmap_name)
    data= HealSparseMap.getValuePixel(HealSparseMap.validPixels)
    # Normalize the colormap
    if vmin is None:
        vmin = np.min(data[data!=HealSparseMap._sentinel])
    if vmax is None:
        vmax = np.max(data[data!=HealSparseMap._sentinel])
    norm = plt.Normalize(vmin, vmax)
    # Create the polygons and transform them with the Geodetic CRS to get them projected in the map without wrapping
    if have_cartopy:
        polycoll = PolyCollection(uv_verts, edgecolor='none', array=data,
                                  norm=norm, transform=cartopy.crs.Geodetic())
    else:
        polycoll = PolyCollection(uv_verts, edgecolor='none', array=data,
                                  norm=norm)
    polycoll.set_cmap(cmap)
    fig.colorbar(polycoll, ax=ax, orientation='vertical', label=colorlabel)
    # Add the PolyCollection to the axes
    ax.add_collection(polycoll, autolim=False)
    if extent is not None:
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
    # Add the grid lines and labels
    if projection in ['PlateCarree','Mercator']:
        # This draws the labels in the ticks for these projections
        ax.gridlines(xlocs=xticks, ylocs=yticks, draw_labels=True)
    else:
        # The ticks are not implemented for other projections so we prepare a workaround
        # This is needed in order to add the ticks.
        try:
            fig.canvas.draw()
            ax.gridlines(xlocs=xticks,ylocs=yticks)
            ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
            ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
            custom_xticks(ax, xticks)
            custom_yticks(ax, yticks)
        except ImportError:
            print('Shapely not found, ticks only implemented for Mercator and PlateCarree projections')
        
    ax.set_xlabel('RA [deg]', fontsize=16)
    ax.set_ylabel('Dec [deg]', fontsize=16)
    if show_coverage:
        # Note: We can generalize this to other projections if needed but, for now, we'll stick to mollview
        plt.axes(ax_cov)
        hp.mollview(HealSparseMap.coverageMask, title='Coverage Mask', hold=True) #coverageMask is a healpix map
        
    if title is not None:
        fig.suptitle(title, fontsize=20, ha='center')
    if savename is not None:
        fig.savefig(savename) 
    fig.show()    
        
