"""
"Magic Duck" for Gaia data tables;
it should also be applicable to other astronomical catalogues
or scientific data tables in general.
It's based on a DuckDB persistent database file that must have
been prepared beforehand.
Jordi Portell i de Mora (jportell@icc.ub.edu)
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.coordinates import SkyCoord
import astropy.units as u
import math as m
import datashader as ds
from datashader import transfer_functions as tf


class GaiaMagicDuck:
    """
    "Magic Duck" for Gaia data tables.
    Jordi Portell (jportell@icc.ub.edu), 2025

    This class provides functions to easily query huge tables
    using DuckDB and then generate statistics, plots or skymaps.
    It should also be applicable to other astronomical catalogues
    or scientific data tables in general.
    It's based on a DuckDB persistent database file that must have
    been prepared beforehand.

    Attributes
    ----------
    dbfile : str
        String with the DuckDB filename (and full path) to be used.
    maintable : str
        Main DB table to be used (e.g. 'gaia_source')
    threads : int, optional
        Number of threads to be used by DuckDB.
    maxram : str, optional
        Maximum RAM (e.g. "'16GB'", beware with the quotation marks) to be used by DuckDB.
    tmpdir : str, optional
        Path to the temporary folder to be used by DuckDB, useful
        for e.g. specifying a fast NVMe disk partition.
    """

    def __init__(self, dbfile, maintable, threads = 2, maxram = "'4GB'", tmpdir = None):
        # Check that we've provided the necessary parameters
        if dbfile is None or maintable is None:
            raise AttributeError("dbfile and maintable must be provided.")
        # Connect to our DB, and show its columns for reference.
        self.con = duckdb.connect(dbfile, read_only=True)
        self.con.sql("set threads = " + str(threads))
        self.con.sql("set memory_limit = " + maxram)
        if tmpdir is not None:
            self.con.sql("set temp_directory = " + tmpdir)
        self.con.sql("PRAGMA enable_progress_bar;")
        print("DuckDB version: " + duckdb.__version__)
        self.maintable = maintable
        self.con.sql("desc " + self.maintable).show(max_rows=300)
        # By default, define 'ra' and 'dec' for the skymap queries
        self.ra = 'ra'
        self.dec = 'dec'
        # Also, indicate that they are not quantized
        self.radecq = 1.0


    def attach(self, dbfile, alias, maintable):
        """
        Attach to an additional DuckDB file identifying it with 'alias'
        and describe the 'maintable' indicated.
        """
        self.con.sql("attach '" + dbfile + "' as " + alias + "(READ_ONLY)")
        self.con.sql("desc " + alias + "." + maintable).show(max_rows=300)


    def quant(self, name, frac):
        """
        Get a string like "(round(name*frac)/frac)", allowing to "quantize"
        some value from a query (good for e.g. histograms to look cleaner)
        """
        return "(round((" + name + ") * " + str(frac) + ") / " + str(frac) + ")"


    def quantlog(self, name, frac):
        """
        Get a string like "(round(log(name)*frac)/frac)", so similar to
        "quant" but already applying a log10
        """
        return "(round(log(" + name + ") * " + str(frac) + ") / " + str(frac) + ")"


    def quantlogoff(self, name, frac, off):
        """
        Get a string like "(round(log(name+off)*frac)/frac)", so similar to
        "quantlog" but adding an offset, in case there can be zeroes or negative values.
        """
        return "(round(log(" + name + "+" + str(off) + ") * " + str(frac) + ") / " + str(frac) + ")"


    def qget(self, sel, cond=None, groupby=None, extra=None, prev=None):
        """
        Run a 'generic' query and get the dataframe (or just show the result, for e.g. "select count").
        Get a string like "select <sel> from <maintable> where <cond> [group by <groupby>]",
        and relay the return (so that we can get e.g. the dataframe).
        'extra' allows to add e.g. "sort by ..."
        'prev' allows to prepend some string, e.g. "COPY (", to e.g. export the result to a CSV
        """
        if (prev != None):
            query = prev + " select " + sel + " from " + self.maintable
        else:
            query = "select " + sel + " from " + self.maintable
        if (cond != None):
            query += " where " + cond
        if (groupby != None):
            query += " group by " + groupby
        if (extra != None):
            query += " " + extra
        # Show the query we've composed, for reference
        print("Running query: " + query)
        return self.con.sql(query)


    def setradec(self, ra, dec, q):
        """
        Set the RA and DEC strings and their quantization (if any)
        ra : string (e.g. 'ra' or 'Alpha')
        dec : string
        q : 1.0 for no quantization; otherwise, something like 10.0 or 12.0
        """
        self.ra = ra
        self.dec = dec
        self.radecq = q


    def qskymap(self, sel, cond):
        """
        Run a query to get a skymap.
        'sel' must indicate what we want to get in each ra/dec bin: "count(*) as counts",
        "MEDIAN(value) as medval", etc.
        'cond' is a standard SQL condition, as in the 'qget' function.
        """
        # If we have the quantized fields:
        if (self.radecq != 1.0):
            query = " select " + self.ra + "/" + str(self.radecq) + " as qra, \
                    " + self.dec + "/" + str(self.radecq) + " as qdec, "
        else:
            query = " select floor(" + self.ra + "*12.0)/12.0 as qra, floor(" + self.dec + "*12.0)/12.0 as qdec, "
        query += sel + " from " + self.maintable
        if (cond != None):
            query += " where " + cond
        query += " group by qra,qdec"
        print("Running query: " + query)
        return self.con.sql(query)


    def _hammer_trf(self, lon, lat):
        """
        Ancillary function to transform coordinates to a Hammer projection,
        for the plotsky() function.
        """
        deno = np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
        x = (2 * np.sqrt(2) * np.cos(lat) * np.sin(lon / 2)) / deno
        y = (np.sqrt(2) * np.sin(lat)) / deno
        return x, y


    def plotsky(self, radec, pmap, reducefunc, binscale, clabel, title,
                cmin = None, cmax = None, cmap = 'turbo', tofile = None,
                doecliptic = False, rot = 0):
        """
        Plot a skymap in Galactic coordinates using Matplotlib.
        It can be a bit slow (around 1 minute depending on the plot size,
        number of bins retrieved from the query, etc.)

        Parameters
        ----------
        radec : dataframe
            dataframe with the 'ra' and 'dec' keys
        pmap : array
            pixel map, i.e. values per ra/dec bin (in principle from the same dataframe), e.g. skymap['counts']
        reducefunc : reduce_C_function
            reduce_C_function to use for the bins, e.g. np.sum or np.median
        binscale : string
            Either "log" for logarithmic bins/scale, or None for linear scale
        clabel : string
            Label for the pixels (for the colorbar), e.g. "Counts"
        title : string
            Plot title
        cmin, cmax : float
            Min/max colormap values to be shown
        cmap : string
            Color map to be used. Some recommended ones: 'turbo', 'Greys_r', 'inferno', 'gist_ncar'...
            (you can add "_r" to reverse the map)
        tofile : string
            You can indicate here a PNG filename to save the figure *instead* of showing it.
        doecliptic : bool
            True to use Ecliptic coordinates, False for Galactic
        rot : int
            You can indicate the degrees that you want to rotate the skymap (just for Galactic, along 'l')
        """
        import matplotlib.style as mplstyle
        print("Mean of values: ", np.mean(pmap.values))
        print("Median of values: ", np.median(pmap.values))
        rse = 0.390152 * ((np.percentile(pmap.values, 90)) - (np.percentile(pmap.values, 10)))
        print("RSE of values: ", rse)
        print("Reducefunc of values: ", reducefunc(pmap.values))
        print("Count of values: ", np.sum(pmap.values))
        print("Getting coordinates...")
        # Get the coordinates and convert them to Galactic
        coords = SkyCoord(ra = radec['qra'].values * u.deg, dec = radec['qdec'].values * u.deg, frame='icrs')
        if (doecliptic):
            newcoords = coords.barycentrictrueecliptic
            l_or_lambda = newcoords.lon.wrap_at(180 * u.deg).radian
            b_or_beta = newcoords.lat.radian
        else:
            newcoords = coords.galactic
            l_shifted = newcoords.l + rot * u.deg
            l_or_lambda = -(l_shifted.wrap_at(180 * u.deg).radian)
            b_or_beta = newcoords.b.radian
        print("Transforming to Hammer...")
        l_rad_tr, b_rad_tr = self._hammer_trf(l_or_lambda, b_or_beta)
        print("Creating figure...")
        # Create the figure with Hammer projection (using the golden ratio)
        plt.figure(figsize=(16.18, 10))
        ax = plt.subplot(111, projection="hammer")
        hb = ax.hexbin(l_rad_tr, b_rad_tr, C = pmap, gridsize=800, cmap=cmap, \
                       reduce_C_function=reducefunc, mincnt=1, bins=binscale, \
                       vmin=cmin, vmax=cmax, antialiased=False)
        cbar = plt.colorbar(hb, orientation="vertical", fraction=0.023, pad=0.03)
        cbar.set_label(clabel + " per 25 arcmin²")
        plt.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.title(title)
        mplstyle.use('fast')
        if (tofile is not None):
            print("Storing to " + tofile + "...")
            plt.savefig(tofile, bbox_inches='tight', pad_inches=0.1)
        else:
            print("Showing...")
            plt.show()
        print("Done!")


    def _hammer_trf_ds(self, lon, lat):
        """
        Ancillary function,
        same as _hammer_trf() but for the datashader-based plotsky.
        """
        deno = np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
        # Clip to avoid division by zero at the poles of the projection
        deno = np.maximum(deno, 1e-9)
        x = (2 * np.cos(lat) * np.sin(lon / 2)) / deno
        y = (np.sin(lat)) / deno
        return x, y


    def plotsky_ds(self, radec, pmap, redfunc='sum', binscale='linear',
                           clabel=None, title=None, cmin=None, cmax=None, cmap='turbo',
                           tofile=None, doecliptic=False, rot=0,
                           raster_width=1600, fig_scale=1.0):
        """
        Plots a skymap in Galactic or Ecliptic coordinates using Datashader for high performance.

        Parameters
        ----------
        radec : DataFrame
            DataFrame with quantized coordinates, e.g., 'qra', 'qdec', and the value to plot.
        pmap : array
            Pixel map, i.e. values per ra/dec bin (in principle from the same dataframe), e.g. skymap['counts']
        redfunc : string, optional
            Reduce function to be used: 'mean' or 'sum' (see https://datashader.org/api.html#reductions)
        binscale : string, optional
            'cbrt' for cube root color scale, 'log' for logarithmic, or 'linear'.
        clabel : string, optional
            Label for the colorbar (e.g. 'source counts'). If not provided, no colorbar will be generated.
        title : string, optional
            Label for the plot title.
        cmin, cmax : float, optional
            Min/max values for the color scale. It will clamp the pmap values before plotting them.
        cmap : string, optional
            Matplotlib colormap name. Some recommended ones: 'turbo', 'Greys_r', 'inferno', 'gist_ncar'...
            (you can add "_r" to reverse the map)
        tofile : string, optional
            PNG filename to save the figure instead of showing it.
        doecliptic : bool, optional
            True for Ecliptic coordinates, False for Galactic.
        rot : float, optional
            Rotation in degrees for Galactic longitude.
        raster_width : int, optional
            The width of the output raster image in pixels. Too high values for too few datapoints will lead to aliasing.
        fig_scale : float, optional
            The figure scale, to get larger (e.g. 2.0) or smaller (e.g. 0.5) plots.
        """

        # Print some info on the map received:
        print("Mean of values: ", np.mean(pmap.values))
        print("Median of values: ", np.median(pmap.values))
        rse = 0.390152 * ((np.percentile(pmap.values, 90)) - (np.percentile(pmap.values, 10)))
        print("RSE of values: ", rse)
        print("Sum of values: ", np.sum(pmap.values))
        print("Getting coordinates...")
        coords = SkyCoord(ra=radec['qra'].values * u.deg,
                        dec=radec['qdec'].values * u.deg,
                        frame='icrs')
        if doecliptic:
            newcoords = coords.barycentrictrueecliptic
            lon = newcoords.lon.wrap_at(180 * u.deg).radian
            lat = newcoords.lat.radian
        else:
            newcoords = coords.galactic
            l_shifted = newcoords.l + rot * u.deg
            # The negation is important for the conventional sky-map orientation
            lon = -(l_shifted.wrap_at(180 * u.deg).radian)
            lat = newcoords.b.radian

        print("Transforming to Hammer projection...")
        hammer_x, hammer_y = self._hammer_trf_ds(lon, lat)

        # --- DATASHADER PIPELINE ---
        print("Preparing plot...")
        # Create a new DataFrame for Datashader with projected coords and values
        plot_df = pd.DataFrame({
            'hammer_x': hammer_x,
            'hammer_y': hammer_y,
            'value': pmap.values
        })
        plot_height = int(raster_width / 1.95)
        # Define the canvas bounds. Hammer projection is roughly within [-2*sqrt(2), 2*sqrt(2)]
        # but we use a slightly smaller range based on typical plots.
        x_range = (-2.05, 2.05)
        y_range = (-1.05, 1.05)
        canvas = ds.Canvas(plot_width=raster_width, plot_height=plot_height,
                        x_range=x_range, y_range=y_range)
        # canvas.points(), instead of canvas.raster(), which allows to
        # correctly handle a DataFrame with x, y, and value columns.
        if redfunc == 'sum':
            agg = canvas.points(plot_df, x='hammer_x', y='hammer_y', agg=ds.sum('value'))
        elif redfunc == 'mean':
            agg = canvas.points(plot_df, x='hammer_x', y='hammer_y', agg=ds.mean('value'))
        # TODO we may add other reduction functions if needed

        # Extract non-NaN values from the aggregated data for auto-ranging
        valid_agg_data = agg.where(np.isfinite(agg)).values.flatten()
        valid_agg_data = valid_agg_data[np.isfinite(valid_agg_data)]
        # Handle auto-ranging for cmin/cmax if not provided
        how = binscale
        if cmin is None:
            if how in ['log', 'cbrt']: # Scales that require positive numbers
                positive_data = valid_agg_data[valid_agg_data > 0]
                cmin = positive_data.min() if len(positive_data) > 0 else 1  # TODO maybe better e.g. 1e-6? To be tested
            else:
                cmin = valid_agg_data.min() if len(valid_agg_data) > 0 else 0
        if cmax is None:
            cmax = valid_agg_data.max() if len(valid_agg_data) > 0 else cmin
        # Generate the shaded plot
        final_cmap = plt.get_cmap(cmap)
        if how == 'cbrt':  # span not supported in cbrt (yet)
            img = tf.shade(agg, cmap=final_cmap, how=how)
        else:
            img = tf.shade(agg, cmap=final_cmap, how=how, span=[cmin,cmax])

        # --- MATPLOTLIB DISPLAY ---
        print("Creating figure...")
        fig_width = 16.18 * fig_scale
        fig_height = 10.0 * fig_scale
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
        ax.imshow(img.to_pil()) # Display the Datashader image
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title, color='black', fontsize=14)  # TODO adjust font size depending on fig width

        # Manually create a colorbar
        if clabel is not None:
            if how == 'log':
                norm = mcolors.LogNorm(vmin=cmin, vmax=cmax)
            elif how == 'cbrt':
                # Use PowerNorm with a gamma of 1/3 for cube-root scaling
                norm = mcolors.PowerNorm(gamma=1./3., vmin=cmin, vmax=cmax)
            else: # 'linear'
                norm = mcolors.Normalize(vmin=cmin, vmax=cmax)
            sm = plt.cm.ScalarMappable(cmap=final_cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.023, pad=0.03)
            cbar.set_label(clabel, color='black')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        fig.tight_layout(pad=0)

        if tofile is not None:
            print(f"Storing to {tofile}...")
            plt.savefig(tofile, bbox_inches='tight', pad_inches=0.1, dpi=150, facecolor='white')
        else:
            print("Showing plot...")
            plt.show()

        print("Done!")


    def qdenmed(self, x, xres, xname, y, yres, yname, cond, logx=False, logy=False, offx=0.0, offy=0.0):
        """
        Run a query (actually 2 queries) to get a density plot plus a running median.
        'cond' is a standard SQL condition, as in the 'query' function.
        It returns the density plot dataframe and the running-median dataframe, in this order.
        """
        # Construct first query (density plot)
        if (logx):
            query = self.quantlogoff(x, xres, offx) + " as " + xname + ", "
        else:
            query = self.quant(x, xres) + " as " + xname + ", "
        if (logy):
            query += self.quantlogoff(y, yres, offy) + " as " + yname + ", "
        else:
            query += self.quant(y, yres) + " as " + yname + ", "
        query += "count(*) as counts"
        # Run first query
        denmap = self.qget(query, cond, groupby = xname + "," + yname).df()
        # Construct second query (running median)
        if (logx):
            query = self.quantlogoff(x, xres, offx) + " as " + xname + ", "
        else:
            query = self.quant(x, xres) + " as " + xname + ", "
        if (logy):
            query += "MEDIAN(" + self.quantlogoff(y, yres, offy) + ") as " + yname + ", "
        else:
            query += "MEDIAN(" + self.quant(y, yres) + ") as " + yname + " "
        # Run second query
        medmap = self.qget(query, cond, groupby = xname, extra = "order by " + xname).df()
        return denmap, medmap


    def plotdenmed(self, denmap, medmap, xname, yname, title, xlabel, ylabel,
                   xrange=None, yrange=None, xstep=None, ystep=None, cmap='turbo'):
        """
        Plot a density map plus its running median.
        """
        plt.figure(figsize=(12, 8))
        h = plt.hexbin(denmap[xname],denmap[yname],C=denmap['counts'], gridsize=400, cmap=cmap,\
                       reduce_C_function=np.sum, bins="log", antialiased=False)
        cbar = plt.colorbar(h, orientation="vertical", fraction=0.04, pad=0.04)
        cbar.set_label("Counts")
        plt.plot(medmap[xname],medmap[yname],'indigo',linewidth=2)
        if xrange!=None:
            plt.xlim(xrange[0], xrange[1])
            if xstep!=None:
                plt.xticks(np.arange(xrange[0], xrange[1], xstep))
        if yrange!=None:
            plt.ylim(yrange[0], yrange[1])
            if ystep!=None:
                plt.yticks(np.arange(yrange[0], yrange[1], ystep))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
