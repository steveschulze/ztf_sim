"""Classes implementing Observing Programs."""

import logging
import numpy as np
import astropy.units as u
from astropy.time import Time
from .constants import EXPOSURE_TIME, READOUT_TIME, TIME_BLOCK_SIZE
from .utils import approx_hours_of_darkness
from .field_selection_functions import *


class ObservingProgram(object):
    """
    A class to represent an observing program.
    Attributes
    ----------
    program_id : int
        The unique identifier for the program.
    subprogram_name : str
        The name of the subprogram.
    program_pi : str
        The principal investigator of the program.
    program_observing_time_fraction : float
        The fraction of observing time allocated to the program.
    subprogram_fraction : float
        The fraction of the program's time allocated to the subprogram.
    field_ids : list or None
        The list of field IDs to be observed, or None if a selection function is used.
    filter_ids : list
        The list of filter IDs to be used.
    internight_gap : float
        The minimum gap between observations on different nights.
    intranight_gap : float
        The minimum gap between observations on the same night.
    n_visits_per_night : int
        The number of visits per night.
    exposure_time : float
        The exposure time for each observation.
    nobs_range : dict or None
        The range of number of observations, or None if not used.
    filter_choice : str
        The method of choosing filters ('rotate' or 'sequence').
    active_months : str or list
        The months during which the program is active ('all' or a list of month numbers).
    field_selection_function : str or None
        The name of the function used to select fields, or None if field_ids are used directly.
    Methods
    -------
    assign_nightly_requests(time, fields, obs_log, other_program_fields, block_programs=False, skymaps=None, **kwargs)
        Assigns nightly observation requests based on the program's parameters.
    time_per_exposure()
        Returns the total time per exposure, including readout time.
    """

    def __init__(self, program_id, subprogram_name, program_pi,
                 program_observing_time_fraction, subprogram_fraction,
                 field_ids, filter_ids, internight_gap, 
                 intranight_gap, n_visits_per_night,
                 exposure_time = EXPOSURE_TIME,
                 nobs_range=None,
                 filter_choice='rotate', 
                 active_months='all',
                 field_selection_function=None):
        """
        Initialize an ObservingProgram instance.

        Parameters:
        program_id (int): Identifier for the program.
        subprogram_name (str): Name of the subprogram.
        program_pi (str): Principal Investigator of the program.
        program_observing_time_fraction (float): Fraction of observing time allocated to the program.
        subprogram_fraction (float): Fraction of the program's time allocated to the subprogram.
        field_ids (list or None): List of field IDs to observe, or None if using a selection function.
        filter_ids (list): List of filter IDs to use.
        internight_gap (float): Minimum gap between observations on different nights.
        intranight_gap (float): Minimum gap between observations on the same night.
        n_visits_per_night (int): Number of visits per night.
        exposure_time (Quantity): Exposure time for each observation.
        nobs_range (tuple or None): Range of number of observations, or None if not specified.
        filter_choice (str): Method for choosing filters ('rotate' or other).
        active_months (str or list): Months when the program is active ('all' or list of months).
        field_selection_function (callable or None): Function to select fields, or None if using field_ids.

        Raises:
        AssertionError: If both field_ids and field_selection_function are provided or both are None.
        """

        assert ((field_ids is None) or (field_selection_function is None))
        assert not((field_ids is None) and (field_selection_function is None))

        self.logger = logging.getLogger(__name__)

        self.program_id = program_id
        self.subprogram_name = subprogram_name
        self.program_pi = program_pi
        self.program_observing_time_fraction = program_observing_time_fraction
        self.subprogram_fraction = subprogram_fraction
        self.field_ids = field_ids
        self.filter_ids = filter_ids

        self.internight_gap = internight_gap
        self.intranight_gap = intranight_gap
        self.n_visits_per_night = n_visits_per_night
        self.exposure_time = exposure_time # a Quantity

        self.nobs_range = nobs_range 
        self.filter_choice = filter_choice
        if active_months != 'all':
            # allow scalar or list input
            self.active_months = np.atleast_1d(active_months)
        else:
            self.active_months = 'all'

        self.field_selection_function = field_selection_function

    def assign_nightly_requests(self, time, fields, obs_log, other_program_fields, block_programs=False, skymaps=None, **kwargs):
        """
        Assigns nightly observation requests based on the given parameters.
        Parameters:
        -----------
        time : astropy.time.Time
            The current time for which the requests are being generated.
        fields : Fields
            An object containing field information and methods to compute blocks and observability.
        obs_log : ObservationLog
            An object containing the observation log.
        other_program_fields : list
            A list of fields from other programs.
        block_programs : bool, optional
            Whether to block other programs (default is False).
        skymaps : optional
            Skymaps to be used for field selection (default is None).
        **kwargs : dict
            Additional keyword arguments.
        Returns:
        --------
        list
            A list of dictionaries containing the request sets for the night.
        """

        # filters are given in filter_ids:
        # either a set of filters, or a fixed sequence
        # filter_choice = 'rotate':
        #   use one filter set per night, keyed by mjd % n_filters
        # filter_choice = 'sequence':
        #   use hard-coded sequence given in filter_ids

        # determine if this program is active this month and return
        # an empty set if not
        if self.active_months != 'all':
            if time.to_datetime().month not in self.active_months:
                return []
        
        
        # compute nightly altaz blocks and observability windows
        fields.compute_blocks(time)
        fields.compute_observability(time)

        n_filters = len(set(self.filter_ids))
        if self.filter_choice == 'rotate':
            night_index_filters = np.floor(time.mjd % n_filters).astype(int)
            filter_ids_tonight = self.filter_ids[night_index_filters]
            filter_ids_last_night = self.filter_ids[night_index_filters - 1]
            # make it a list
            filter_ids_tonight = [filter_ids_tonight]
        else:
            filter_ids_tonight = list(set(self.filter_ids))

        # Choose which fields will be observed

        # minimum time to observe N visits
        obs_field_ids = fields.select_field_ids(observable_hours_range=
            [(self.n_visits_per_night*TIME_BLOCK_SIZE).to(u.hour).value, 24.])

        # if needed, compute the OP fields on a nightly basis
        if self.field_selection_function is not None:
            try:
                selection_function = globals()[self.field_selection_function]
                field_ids = selection_function(time, obs_log, other_program_fields, fields, skymaps)
                self.logger.info(f'Program ID {self.program_id}, subprogram {self.subprogram_name}: selected {len(field_ids)} fields')
                self.logger.debug(f'    {field_ids}')
            except Exception as e:
                #raise(e) # needed to debug filter_selection_functions
                self.logger.exception(e)
                self.logger.warning(f'Error in generating nightly field list for Program ID {self.program_id}, subprogram {self.subprogram_name}, returning zero fields!')  
                return []
        else:
            field_ids = self.field_ids

        # now form the intersection of observable fields and the OP fields
        pool_ids = obs_field_ids.intersection(field_ids)
        self.logger.debug(f'Program ID {self.program_id}, subprogram {self.subprogram_name}: {len(pool_ids)} fields observable')

        # get the times they were last observed:
        # (note that fields *never* observed will not be included)
        # since this is function is for determining requests
        # at the start of the night, exclude observations taken tonight
        # this lets us restart the scheduler without breaking things
        last_observed_times = obs_log.select_last_observed_time_by_field(
                field_ids = pool_ids,
                filter_ids = filter_ids_tonight,
                program_ids = [self.program_id],
                subprogram_names = [self.subprogram_name],
                # arbitrary early date; start of night tonight
                mjd_range = [Time('2001-01-01').mjd,np.floor(time.mjd)])

        # we want an object observed at the end of the night N days ago
        # to be observed at the start of the night now.
        # Max night length is 12.2 hours
        cutoff_time = (time - (self.internight_gap - 0.6 * u.day)).mjd

        # find fields last observed more recently than that
        wrecent = (last_observed_times['expMJD'] >= cutoff_time)
        recent_field_ids = last_observed_times.loc[wrecent].index.tolist()

        if self.field_selection_function is None:
            # reduce the list to only those not recently observed:
            pool_ids_old = [idi for idi in pool_ids if idi not in recent_field_ids]
            request_fields = fields.fields.loc[pool_ids_old]
        else:
            # field_selection_function needs to apply the cadence cut
            request_fields = fields.fields.loc[pool_ids]

        # if we have an nobs_range argument (eg for reference building), use it
        if self.nobs_range is not None:
            if 'program_ids' not in self.nobs_range:
                program_ids = None
            else:
                program_ids = self.nobs_range['program_ids']

            if 'subprogram_names' not in self.nobs_range:
                subprogram_names = None
            else:
                subprogram_names = self.nobs_range['subprogram_names']

            if 'filter_id' in self.nobs_range:
                self.nobs_range['filter_ids'] = [self.nobs_range['filter_id']]
            if 'filter_ids' not in self.nobs_range:
                filter_ids = None
            else:
                filter_ids = self.nobs_range['filter_ids'] 
                
            if 'mjd_range' not in self.nobs_range:
                mjd_range = None
            else:
                mjd_range = self.nobs_range['mjd_range']

            assert 'min_obs' in self.nobs_range
            assert 'max_obs' in self.nobs_range
                
            nobs = obs_log.select_n_obs_by_field(filter_ids = filter_ids,
                    program_ids = program_ids, 
                    subprogram_names = subprogram_names,
                    mjd_range = mjd_range)
            
            # function above only returns fields that have been observed at
            # least once.  use the intersection if min_obs > 0:
            w = ((nobs >= self.nobs_range['min_obs']) & 
                    (nobs <= self.nobs_range['max_obs']))
            if self.nobs_range['min_obs'] > 0:
                nobs_inrange = nobs.loc[w]
                request_fields = request_fields.join(nobs_inrange,how='inner')
            else:
                # drop rows out of range (which here means only those with 
                # nobs > max_obs
                nobs_outofrange = nobs.loc[~w]
                # find fields that are in request_fields but out of range
                nobs_outofrange = request_fields.join(nobs_outofrange,how='inner')
                # now drop them
                request_fields = request_fields.drop(nobs_outofrange.index)
            

        # construct request sets: list of inputs to RequestPool.add_requests
        # scalar everything except field_ids

        if self.filter_choice == 'rotate':
            filter_sequence = [filter_ids_tonight[0] for i in
                               range(self.n_visits_per_night)]
        elif self.filter_choice == 'sequence':
            assert(len(self.filter_ids) == self.n_visits_per_night)
            filter_sequence = self.filter_ids.copy()

        self.logger.debug(f'Program ID {self.program_id}, subprogram {self.subprogram_name}: {len(request_fields.index.values)} fields requested')

        request_set = []
        request_set.append(
            {'program_id': self.program_id,
             'subprogram_name': self.subprogram_name,
             'program_pi': self.program_pi,
             'field_ids': request_fields.index.values,
             'filter_ids': filter_sequence,
             'exposure_time': self.exposure_time,
             'intranight_gap': self.intranight_gap,
             'total_requests_tonight': self.n_visits_per_night})

        return request_set

    def time_per_exposure(self):
        """
        Calculate the total time per exposure.

        This method returns the sum of the exposure time and the readout time.

        Returns:
            float: The total time per exposure.
        """
        return self.exposure_time + READOUT_TIME
