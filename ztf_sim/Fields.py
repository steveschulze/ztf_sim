"""Routines for working with the ZTF discrete field grid"""
from __future__ import absolute_import

from builtins import zip
from builtins import object
import numpy as np
import pandas as pd
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time
from collections import defaultdict
import itertools
from .utils import *
from .constants import BASE_DIR, P48_loc, PROGRAM_IDS, FILTER_IDS
from .constants import TIME_BLOCK_SIZE, MAX_AIRMASS, EXPOSURE_TIME


class Fields(object):
    """Class for accessing field grid."""
    # TODO: consider using some of PTFFields.py code

    def __init__(self, field_filename=BASE_DIR + '../data/ZTF_Fields.txt'):
        self._load_fields(field_filename)
        self.loc = P48_loc
        self.current_block_night_mjd = None  # np.floor(time.mjd)
        self.current_blocks = None
        self.block_alt = None
        self.block_az = None
        self.observable_hours = None

    def _load_fields(self, field_filename):
        """Loads a field grid of the format generated by Tom B.
        Expects field_id, ra (deg), dec (deg) columns"""

        df = pd.read_table(field_filename,
            names=['field_id','ra','dec','ebv','l','b',
                'ecliptic_lon', 'ecliptic_lat', 'number'],
            sep='\s+',usecols=['field_id','ra','dec', 'l','b', 
                'ecliptic_lon', 'ecliptic_lat'],index_col='field_id',
            skiprows=1)


        # drop fields below dec of -31 degrees for speed
        # (grid_id = 0 has a row at -31.5)
        df = df[df['dec'] >= -32]

        # label the grid ids
        grid_id_boundaries = \
            {0: {'min':1,'max':999},
             1: {'min':1001,'max':1999},
             2: {'min':2001,'max':2999},
             3: {'min':3001,'max':3999}}

        # intialize with a bad int value
        df['grid_id'] = 99

        for grid_id, bounds in list(grid_id_boundaries.items()):
            w = (df.index >= bounds['min']) &  \
                    (df.index <= bounds['max'])
            df.loc[w,'grid_id'] = grid_id

        # initialize the last observed time
        # TODO: load last observed time per filter & program

        for program_id in PROGRAM_IDS:
            for filter_id in FILTER_IDS:
                df['last_observed_{}_{}'.format(program_id, filter_id)] = \
                    Time('2001-01-01').mjd
                df['first_obs_tonight_{}_{}'.format(program_id, filter_id)] = \
                    np.nan

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

    def compute_blocks(self, time, time_block_size=TIME_BLOCK_SIZE):
        """Store alt/az for tonight in blocks"""

        # TODO: does this really belong in Fields, since it changes night to
        # night?

        # check if we've already computed for tonight:
        block_night = np.floor(time.mjd).astype(np.int)
        if self.current_block_night_mjd == block_night:
            return

        self.current_block_night_mjd = block_night

        blocks, times = nightly_blocks(time, time_block_size=time_block_size)
        self.current_blocks = blocks

        alt_blocks = {}
        az_blocks = {}
        for bi, ti in zip(blocks, times):
            altaz = self.alt_az(ti)
            alt_blocks[bi] = altaz.alt
            az_blocks[bi] = altaz.az

        # DataFrames indexed by field_id, columns are block numbers
        self.block_alt = pd.DataFrame(alt_blocks)
        self.block_az = pd.DataFrame(az_blocks)

        block_airmass = altitude_to_airmass(self.block_alt)
        w = (block_airmass <= MAX_AIRMASS) & (block_airmass >= 1.0)
        # average airmass over the time we're above MAX_AIRMASS
        mean_observable_airmass = block_airmass[w].mean(axis=1)
        mean_observable_airmass.name = 'mean_observable_airmass'
        self.mean_observable_airmass = mean_observable_airmass

    def compute_observability(self, max_airmass=MAX_AIRMASS,
                              time_block_size=TIME_BLOCK_SIZE):
        """For each field_id, use the number of nighttime blocks above max_airmass to compute observability time."""
        # TODO: not being careful about retaining what the airmass limit is
        # once the values are stored

        min_alt = airmass_to_altitude(max_airmass)

        observable_hours = (self.block_alt >= min_alt.to(u.degree).value).sum(axis=1) * \
            (TIME_BLOCK_SIZE.to(u.hour))
        observable_hours.name = 'observable_hours'
        self.observable_hours = observable_hours

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
        for axis in ['dome', 'dec', 'ha']:
            if axis == 'dome':
                current_coord = current_state['current_domeaz'].value
            if axis == 'ha':
                # convert to RA for ease of subtraction
                current_coord = HA_to_RA(current_state['current_ha'],
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

        return dfmax, df_altaz

    def select_fields(self,
                      ra_range=None, dec_range=None,
                      l_range=None, b_range=None,
                      abs_b_range=None,
                      ecliptic_lon_range=None, ecliptic_lat_range=None,
                      grid_id=None,
                      program_id=None, filter_id=None,
                      n_obs_range=None, last_observed_range=None,
                      observable_hours_range=None,
                      reducefunc=None):
        """Select a subset of fields based on their sky positions.

        Each _range keyword takes a list[min, max].
        grid_id is a scalar
        n_obs_range and last_observed_range require specifying
        program_id and filter_id.  if these are not scalar,
        reducefunc (e.g., [np.min, np.max]) is used to collapse the range
        selection over multiple programs or filters
        reducefunc should have two elements to describe the reduction for the
        min comparison and the max.

        Returns a boolean array indexed by field_id."""

        # start with a boolean True series:
        cuts = (self.fields['ra'] == self.fields['ra'])

        if observable_hours_range is not None:
            # check that we've computed observable_hours
            assert(self.observable_hours is not None)
            fields = self.fields.join(self.observable_hours)
        else:
            fields = self.fields

        range_keys = ['ra', 'dec', 'l', 'b', 'ecliptic_lon', 'ecliptic_lat',
                      'observable_hours']

        assert((b_range is None) or (abs_b_range is None))

        for i, arg in enumerate([ra_range, dec_range, l_range, b_range,
                                 ecliptic_lon_range, ecliptic_lat_range,
                                 observable_hours_range]):
            if arg is not None:
                cuts = cuts & (fields[range_keys[i]] >= arg[0]) & \
                    (fields[range_keys[i]] <= arg[1])

        # easier cuts for Galactic/Extragalactic
        if abs_b_range is not None:
            cuts = cuts & (np.abs(fields['b']) >= abs_b_range[0]) & \
                (np.abs(fields['b']) <= abs_b_range[1])

        scalar_keys = ['grid_id']

        for i, arg in enumerate([grid_id]):
            if arg is not None:
                cuts = cuts & (fields[scalar_keys[i]] == arg)

        # n_obs and last_observed require special treatment,
        # since we have to specify the program_id and filter_id

        pf_keys = ['n_obs', 'last_observed']

        for i, arg in enumerate([n_obs_range, last_observed_range]):
            if arg is not None:
                assert ((program_id is not None) and (filter_id is not None))
                if (scalar_len(program_id) == 1) and (scalar_len(filter_id) == 1):
                    key = '{}_{}_{}'.format(pf_keys[i], program_id, filter_id)
                    cuts = cuts & (fields[key] >= arg[0]) & \
                        (fields[key] <= arg[1])
                else:
                    # combined selections across several filters
                    # and/or programs
                    assert (reducefunc is not None)
                    assert len(reducefunc) == 2
                    # build the keys
                    keys = []
                    for pi in np.atleast_1d(program_id):
                        for fi in np.atleast_1d(filter_id):
                            keys.append('{}_{}_{}'.format(pf_keys[i], pi, fi))
                    mincomp = fields[keys].apply(reducefunc[0], axis=1)
                    maxcomp = fields[keys].apply(reducefunc[1], axis=1)
                    cuts = cuts & (mincomp >= arg[0]) & \
                        (maxcomp <= arg[1])

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

        # need this syntax to avoid setting on a copy
        self.fields.loc[field_id,
                        'last_observed_{}_{}'.format(program_id, filter_id)] = time_obs.mjd

        if np.isnan(self.fields.loc[field_id,
                                    'first_obs_tonight_{}_{}'.format(
                                        program_id, filter_id)]):
            self.fields.loc[field_id,
                            'first_obs_tonight_{}_{}'.format(
                                program_id, filter_id)] = time_obs.mjd

        self.fields.loc[field_id,
                        'n_obs_{}_{}'.format(program_id, filter_id)] += 1

    def count_total_obs_by_program(self):
        """Sum total number of exposures by program id"""

        count = defaultdict(int)
        for program_id in PROGRAM_IDS:
            cols = ['n_obs_{}_{}'.format(program_id, filter_id)
                    for filter_id in FILTER_IDS]
            for col in cols:
                count[program_id] += self.fields[col].sum()

        return count

    def clear_first_obs(self):
        """Reset the time of the nightly first observations."""

        for program_id in PROGRAM_IDS:
            for filter_id in FILTER_IDS:
                self.fields.loc[:, 'first_obs_tonight_{}_{}'.format(
                    program_id, filter_id)] = np.nan


def generate_test_field_grid(filename=BASE_DIR+'../data/ZTF_fields.txt',
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