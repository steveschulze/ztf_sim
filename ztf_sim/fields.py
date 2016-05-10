"""Routines for working with the ZTF discrete field grid"""

import numpy as np
import pandas as pd
from utils import *
from constants import *
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time
from collections import defaultdict
import itertools


class Fields(object):
    """Class for accessing field grid."""
    # TODO: consider using some of PTFFields.py code

    def __init__(self, dbname='test_fields'):
        self._load_fields(dbname=dbname)
        self.loc = P48_loc
        self.current_blocks = None
        self.block_alt = None
        self.block_az = None
        self.observable_hours = None

    def _load_fields(self, dbname='test_fields'):
        """Loads a field grid from ../data/{dbname}.db.  
        Expects field_id, ra (deg), dec (deg) columns"""
        df = df_read_from_sqlite(dbname, index_col='field_id')

        # drop fields below dec of -30 degrees for speed
        df = df[df['dec'] >= -30]

        # initialize the last observed time
        # TODO: load last observed time per filter & program

        for program_id in PROGRAM_IDS:
            for filter_id in FILTER_IDS:
                df['last_observed_{}_{}'.format(program_id, filter_id)] = \
                    Time('2001-01-01')

        # TODO: load total observations per filter & program

        for program_id in PROGRAM_IDS:
            for filter_id in FILTER_IDS:
                df['n_obs_{}_{}'.format(program_id, filter_id)] = 0

        self.fields = df
        self.field_coords = self._field_coords()

    def _field_coords(self, cuts=None):
        """Generate an astropy SkyCoord object for current fields"""
        if cuts is None:
            fields = self.fields
        else:
            fields = self.fields[cuts]
        return coord.SkyCoord(fields['ra'],
                              fields['dec'], frame='icrs', unit='deg')

    def alt_az(self, time, cuts=None):
        """return Altitude & Azimuth by field at a given time"""

        if cuts is None:
            index = self.fields.index
            fieldsAltAz = self.field_coords.transform_to(
                coord.AltAz(obstime=time, location=self.loc))
        else:
            # warning: specifying cuts makes this much slower
            index = self.fields[cuts].index
            fieldsAltAz = self._field_coords(cuts=cuts).transform_to(
                coord.AltAz(obstime=time, location=self.loc))

        return pd.DataFrame({'alt': fieldsAltAz.alt, 'az': fieldsAltAz.az},
                            index=index)

    def overhead_time(self, current_state, cuts=None):
        """Calculate overhead time in seconds from current position.
        Also returns current altitude, for convenience.

        cuts is a boolean series indexed by field_id, as generated by 
        select_fields """
        # TODO: think about partitioning this. dome slew is the only
        # time-dependent value
        # TODO: is block-sized discretization accurate enough?
        # TODO: figure out appropriate treatment of dome at zenith

        if cuts is None:
            fields = self.fields
        else:
            fields = self.fields[cuts]

        df_altaz = self.alt_az(current_state['current_time'], cuts=cuts)
        df = fields.join(df_altaz)

        slews_by_axis = {'readout': READOUT_TIME}
        for axis in ['ha', 'dec', 'dome']:
            if axis == 'dome':
                current_coord = current_state['current_domeaz'].value
            if axis == 'ha':
                # convert to RA for ease of subtraction
                current_coord = RA_to_HA(current_state['current_ha'],
                                         current_state['current_time']).degree
            if axis == 'dec':
                current_coord = current_state['current_dec'].value
            coord = P48_slew_pars[axis]['coord']
            dangle = np.abs(df[coord] - current_coord)
            angle = np.where(dangle < (360. - dangle), dangle, 360. - dangle)
            slews_by_axis[axis] = slew_time(axis, angle * u.deg)

        dfslews = pd.DataFrame(slews_by_axis, index=df.index)

        dfmax = dfslews.max(axis=1)
        dfmax = pd.DataFrame(dfmax)
        dfmax.columns = ['overhead_time']

        return dfmax, df_altaz['alt']

    def select_fields(self,
                      ra_range=None, dec_range=None,
                      l_range=None, b_range=None,
                      ecliptic_lon_range=None, ecliptic_lat_range=None,
                      grid_id=None,
                      program_id=None, filter_id=None,
                      n_obs_range=None, last_observed_range=None):
        """Select a subset of fields based on their sky positions.

        Each _range keyword takes a list [min,max].
        grid_id is a scalar
        (for now), n_obs_range and last_observed_range require specifying
        program_id and filter_id.

        Returns a boolean array indexed by field_id."""

        # start with a boolean True series:
        cuts = (self.fields['ra'] == self.fields['ra'])

        range_keys = ['ra', 'dec', 'l', 'b', 'ecliptic_lon', 'ecliptic_lat']

        for i, arg in enumerate([ra_range, dec_range, l_range, b_range,
                                 ecliptic_lon_range, ecliptic_lat_range]):
            if arg is not None:
                cuts = cuts & (self.fields[range_keys[i]] >= arg[0]) & \
                    (self.fields[range_keys[i]] <= arg[1])

        scalar_keys = ['grid_id']

        for i, arg in enumerate([grid_id]):
            if arg is not None:
                cuts = cuts & (self.fields[scalar_keys[i]] == arg)

        # n_obs and last_observed require special treatment,
        # since we have to specify the program_id and filter_id

        pf_keys = ['n_obs', 'last_observed']

        for i, arg in enumerate([n_obs_range, last_observed_range]):
            if arg is not None:
                assert ((program_id is not None) and (filter_id is not None))
                # TODO: allow combined selections across several filters
                # and/or programs
                key = '{}_{}_{}'.format(pf_keys[i], program_id, filter_id)
                cuts = cuts & (self.fields[key] >= arg[0]) & \
                    (self.fields[key] <= arg[1])

        return cuts

    def select_field_ids(self, **kwargs):
        """Returns a pandas index"""
        cuts = self.select_fields(**kwargs)
        return self.fields[cuts].index

    def mark_field_observed(self, request, current_state):
        """Update time last observed and number of observations for a single field"""

        field_id = request['target_field_id']
        program_id = request['target_program_id']
        filter_id = request['target_filter_id']
        time_obs = current_state['current_time'] - EXPOSURE_TIME

        self.fields.ix[field_id]['last_observed_{}_{}'.format(program_id, filter_id)] = \
            time_obs

        self.fields.ix[field_id][
            'n_obs_{}_{}'.format(program_id, filter_id)] += 1


def generate_test_field_grid(filename='../data/ZTF_fields.txt',
                             dbname='test_fields'):
    """Convert Eran's field grid to sqlite"""

    df = pd.read_table(filename, delimiter='\s+', skiprows=1,
                       names=['field_id', 'ra', 'dec', 'extinction_b-v',
                              'l', 'b',
                              'ecliptic_lon', 'ecliptic_lat'], index_col=0)

    # insert label for offset grids
    grid = pd.Series(df.index >=
                     1000, index=df.index, name='grid_id', dtype=np.int8)

    df = df.join(grid)

    df_write_to_sqlite(df[['ra', 'dec', 'l', 'b', 'ecliptic_lon', 'ecliptic_lat',
                           'grid_id']], dbname, index_label='field_id')
