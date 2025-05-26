"""
"Magic Duck" for Gaia data tables;
it should also be applicable to other astronomical catalogues
or scientific data tables in general.
It's based on a DuckDB persistent database file that must have
been prepared beforehand.
Jordi Portell (jportell@icc.ub.edu)
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import math as m


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
    threads : int
        Number of threads to be used by DuckDB.
    maxram : str
        Maximum RAM (e.g. '16GB') to be used by DuckDB.
    tmpdir : str
        Path to the temporary folder to be used by DuckDB, useful
        for e.g. specifying a fast NVMe disk partition.
    """

    def __init__(self, dbfile, maintable, threads = 4, maxram = '4GB', tmpdir = None):
        # Check that we've provided the necessary parameters
        if dbfile is None or maintable is None:
            raise AttributeError("dbfile and maintable must be provided.")
        # Connect to our DB, and show its columns for reference.
        self.con = duckdb.connect(dbfile, read_only=True)
        self.con.sql("set threads = " + str(threads))
        self.con.sql("set memory_limit = " + maxram)
        if tmpdir is not None:
            self.con.sql("set temp_directory = " + tmpdir)
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


    def _hammer_trf(self, lon, lat):
        """
        Ancillary function to transform coordinates to a Hammer projection
        """
        deno = np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
        x = (2 * np.sqrt(2) * np.cos(lat) * np.sin(lon / 2)) / deno
        y = (np.sqrt(2) * np.sin(lat)) / deno
        return x, y


    def quant(self, name, frac):
        """
        Get a string like "(round(name*frac)/frac)"
        """
        return "(round((" + name + ") * " + str(frac) + ") / " + str(frac) + ")"


    def quantlog(self, name, frac):
        """
        Get a string like "(round(log(name)*frac)/frac)"
        """
        return "(round(log(" + name + ") * " + str(frac) + ") / " + str(frac) + ")"


    def quantlogoff(self, name, frac, off):
        """
        Get a string like "(round(log(name+off)*frac)/frac)"
        """
        return "(round(log(" + name + "+" + str(off) + ") * " + str(frac) + ") / " + str(frac) + ")"


    def qget(self, sel, cond, groupby=None, extra=None, prev=None):
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
        'cond' is a standard SQL condition, as in the 'query' function.
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


    def plotsky(self, radec, pmap, reducefunc, binscale, clabel, title,
                cmin = None, cmax = None, cmap = 'turbo', tofile = None,
                doecliptic = False, rot = 0):
        """
        Plot a skymap in Galactic coordinates.
        
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
        rse = 0.390152 * ((np.percentile(pmap.values, 0.9)) - (np.percentile(pmap.values, 0.1)))
        print("RSE of values: ", rse)
        print("Reducefunc of values: ", reducefunc(pmap.values))
        print("Count of values: ", np.sum(pmap.values))
        print("Getting coords...")
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
