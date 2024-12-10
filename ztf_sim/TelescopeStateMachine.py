"""State machine for simulating observations."""

from transitions import Machine
from astropy.time import Time
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
import logging
from .utils import *
from .constants import BASE_DIR, P48_loc, FILTER_IDS
from .constants import READOUT_TIME, EXPOSURE_TIME, FILTER_CHANGE_TIME, slew_time

class TelescopeStateMachine(Machine):
    """
    A state machine to manage the operations of a telescope, including slewing, changing filters, and exposing.

    Attributes:
        current_time (Time): The current time in UTC.
        current_ha (Angle): The current hour angle of the telescope.
        current_dec (Angle): The current declination of the telescope.
        current_domeaz (Angle): The current azimuth of the telescope dome.
        current_filter_id (int): The ID of the current filter in use.
        filters (list): A list of available filter IDs.
        current_zenith_seeing (Angle): The current seeing at the zenith.
        target_skycoord (SkyCoord): The target sky coordinates for the telescope.
        historical_observability_year (int): The year for historical observability data.
        observability (PTFObservabilityDB): The observability database.
        logger (Logger): Logger for the state machine.

    Methods:
        current_state_dict(): Return current state parameters in a dictionary.
        can_observe(): Check if the telescope can observe based on time and weather.
        slew_allowed(target_skycoord): Check if the slew to the target coordinates is within allowed limits.
        process_slew(target_skycoord, readout_time=READOUT_TIME): Process the slewing of the telescope to the target coordinates.
        process_filter_change(target_filter_id, filter_change_time=FILTER_CHANGE_TIME): Process the changing of the filter.
        process_exposure(exposure_time): Process the exposure of the telescope.
        wait(wait_time=EXPOSURE_TIME): Wait for a specified amount of time.
    """

    def __init__(self, current_time=Time('2018-01-01', scale='utc',
                                         location=P48_loc),
                 current_ha=0. * u.deg, current_dec=33.36 * u.deg,
                 current_domeaz=180. * u.deg,
                 current_filter_id=2, filters=FILTER_IDS,
                 current_zenith_seeing=2.0 * u.arcsec,
                 target_skycoord=None,
                 historical_observability_year=2015):

        """
        Initialize the TelescopeStateMachine.

        Parameters
        ----------
        current_time : astropy.time.Time, optional
            The current time, default is '2018-01-01' UTC.
        current_ha : astropy.units.Quantity, optional
            The current hour angle, default is 0 degrees.
        current_dec : astropy.units.Quantity, optional
            The current declination, default is 33.36 degrees.
        current_domeaz : astropy.units.Quantity, optional
            The current dome azimuth, default is 180 degrees.
        current_filter_id : int, optional
            The current filter ID, default is 2.
        filters : list, optional
            The list of available filter IDs, default is FILTER_IDS.
        current_zenith_seeing : astropy.units.Quantity, optional
            The current zenith seeing, default is 2.0 arcseconds.
        target_skycoord : astropy.coordinates.SkyCoord, optional
            The target sky coordinates, default is None.
        historical_observability_year : int, optional
            The year for historical observability data, default is 2015.
        """

        # Define some states.
        states = ['ready', 'cant_observe',
                  'slewing', 'changing_filters', 'exposing']

        # define the transitions

        transitions = [
            {'trigger': 'start_slew', 'source': 'ready', 'dest': 'slewing',
                'after': ['process_slew', 'stop_slew'],
                'conditions': 'slew_allowed'},
            {'trigger': 'stop_slew', 'source': 'slewing', 'dest': 'ready'},
            # for now do not require filter changes to include a slew....
            {'trigger': 'start_filter_change', 'source': 'ready',
                'dest': 'changing_filters',
                'after': ['process_filter_change', 'stop_filter_change']},
            {'trigger': 'stop_filter_change', 'source': 'changing_filters',
                'dest': 'ready'},
            {'trigger': 'start_exposing', 'source': 'ready', 'dest': 'exposing',
                'after': ['process_exposure', 'stop_exposing']},
            {'trigger': 'stop_exposing', 'source': 'exposing', 'dest': 'ready'},
            # I would like to automatically set the cant_observe state from
            # start_exposing, but that doesn't seem to work.
            {'trigger': 'check_if_ready', 'source': ['ready', 'cant_observe'],
                'dest': 'ready', 'conditions': 'can_observe'},
            {'trigger': 'set_cant_observe', 'source': '*',
                'dest': 'cant_observe'}
        ]

        # Initialize the state machine.  syntax from
        # https://github.com/tyarkoni/transitions
        Machine.__init__(self, states=states,
                         transitions=transitions,
                         initial='ready')

        self.current_time = current_time
        self.current_ha = current_ha
        self.current_dec = current_dec
        self.current_domeaz = current_domeaz
        self.current_filter_id = current_filter_id
        self.filters = filters
        self.current_zenith_seeing = current_zenith_seeing
        self.target_skycoord = target_skycoord

        # historical observability
        self.historical_observability_year = historical_observability_year
        self.observability = PTFObservabilityDB()

        self.logger = logging.getLogger(__name__)
        #self.logger = logging.getLogger('transitions')

    def current_state_dict(self):
        """Return current state parameters in a dictionary"""
        return {'current_time': self.current_time,
                'current_ha': self.current_ha,
                'current_dec': self.current_dec,
                'current_domeaz': self.current_domeaz,
                'current_filter_id': self.current_filter_id,
                'current_zenith_seeing': self.current_zenith_seeing,
                'filters': self.filters,
                'target_skycoord': self.target_skycoord}

    def can_observe(self):
        """Check for night and weather"""
        self.logger.info(self.current_time.iso)

        # start by checking for 12 degree twilight
        if coord.get_sun(self.current_time).transform_to(
                coord.AltAz(obstime=self.current_time,
                            location=P48_loc)).alt.is_within_bounds(
                upper=-12. * u.deg):
            if self.historical_observability_year is None:
                # don't use weather, just use 12 degree twilight
                return True
            else:
                is_observable = self.observability.check_historical_observability(
                    self.current_time, year=self.historical_observability_year)
                if not is_observable:
                    # optimization: fast-forward to start of next block
                    block_now = block_index(self.current_time)
                    block_end_time = block_index_to_time(block_now,
                        self.current_time, where='end')[0]
                    self.logger.info('Weathered out.  Fast forwarding to end of this block: {}'.format(
                        block_end_time.iso))
                    self.current_time = block_end_time

                return is_observable
        else:
            # daytime
            # optimization: fast-forward to sunset
            next_twilight = next_12deg_evening_twilight(self.current_time)
            self.logger.info('Fast forwarding to 12 deg twilight: {}'.format(
                next_twilight.iso))
            self.current_time = next_twilight
            return False

    def slew_allowed(self, target_skycoord):
        """Check that slew is within allowed limits"""

        if (skycoord_to_altaz(target_skycoord, self.current_time).alt
            < (10. * u.deg)):
            return False

        if ((target_skycoord.dec < -35. * u.deg) or
                (target_skycoord.dec > 90. * u.deg)):
            return False
        return True

    def process_slew(self, target_skycoord, readout_time=READOUT_TIME):
        """
        Processes the telescope slewing to a new target sky coordinate.

        Parameters:
        target_skycoord (SkyCoord): The target sky coordinates to slew to.
        readout_time (Quantity, optional): The readout time during the slew. 
                                           Defaults to READOUT_TIME.

        Notes:
        - If readout_time is nonzero, it is assumed that the telescope is reading during the slew,
          which sets the lower limit for the time between exposures.
        - The function calculates the time required to slew to the new target coordinates.
        - The current time is updated based on the calculated slew time.
        - The function updates the current hour angle (HA), declination (dec), and dome azimuth (domeaz)
          after the slew is complete.
        """

        # if readout_time is nonzero, assume we are reading during the slew,
        # which sets the lower limit for the time between exposures.

        self.target_skycoord = target_skycoord

        target_ha = RA_to_HA(self.target_skycoord.ra, self.current_time)
        target_domeaz = skycoord_to_altaz(self.target_skycoord,
                                          self.current_time).az
        target_dec = target_skycoord.dec

        # calculate time required to slew
        # duplicates codes in fields.py--consider refactoring
        axis_slew_times = [READOUT_TIME]
        for axis in ['ha', 'dec', 'domeaz']:
            dangle = np.abs(eval("target_{}".format(axis)) -
                            eval("self.current_{}".format(axis)))
            angle = np.where(dangle < (360. * u.deg - dangle), dangle,
                             360. * u.deg - dangle)
            axis_slew_times.append(slew_time(axis[:4], angle))

        net_slew_time = np.max([st.value for st in axis_slew_times]) *\
            axis_slew_times[0].unit

        # update the time
        self.current_time += net_slew_time
        # small deviation here: ha, az of target ra shifts (usually!)
        # modestly during slew,
        # so store the value after the slew is complete.

        target_ha = RA_to_HA(self.target_skycoord.ra, self.current_time)
        target_domeaz = skycoord_to_altaz(self.target_skycoord,
                                          self.current_time).az
        self.current_ha = target_ha
        self.current_dec = self.target_skycoord.dec
        self.current_domeaz = target_domeaz

    def process_filter_change(self, target_filter_id,
                              filter_change_time=FILTER_CHANGE_TIME):
        """
        Processes the change of the telescope's filter.

        Parameters:
        target_filter_id (int): The ID of the target filter to change to.
        filter_change_time (float, optional): The time it takes to change the filter. Defaults to FILTER_CHANGE_TIME.

        Updates:
        self.current_filter_id: Sets to the target_filter_id.
        self.current_time: Increments by filter_change_time if the filter is changed.
        """
        if self.current_filter_id != target_filter_id:
            self.current_filter_id = target_filter_id
            self.current_time += filter_change_time

    def process_exposure(self, exposure_time):
        # annoyingly, transitions doesn't let me modify object
        # variables in the trigger functions themselves
        self.current_time += exposure_time
        # update ha and domeaz for tracking during the exposure
        target_ha = RA_to_HA(self.target_skycoord.ra, self.current_time)
        target_domeaz = skycoord_to_altaz(self.target_skycoord,
                                          self.current_time).az
        self.current_ha = target_ha
        self.current_domeaz = target_domeaz

    def wait(self, wait_time=EXPOSURE_TIME):
        self.current_time += wait_time


class PTFObservabilityDB(object):
    """
    PTFObservabilityDB is a class that provides methods to check the historical observability of the Palomar Transient Factory (PTF) based on weather data.

    Methods
    -------
    __init__():
        Initializes the PTFObservabilityDB instance by reading weather data from an SQLite database and setting it as a DataFrame indexed by year and block.

    check_historical_observability(time, year=2015, nobs_min=5):
        Given a (possibly future) UTC time, looks up whether PTF was observing at that time in the specified year.

        year : int, optional
            Year to check PTF historical observing (default is 2015). Valid range is from 2009 to 2015.
        nobs_min : int, optional
            Minimum number of observations per block to count as observable (default is 5).

        -------
        bool
            True if the number of observations in the specified block is greater than or equal to nobs_min, False otherwise.
    """

    def __init__(self):
        """
        Initializes the TelescopeStateMachine instance.

        This constructor reads data from an SQLite database table named 'weather_blocks'
        and sets the DataFrame index to ['year', 'block'].

        Attributes:
            df (pd.DataFrame): A DataFrame containing the weather blocks data with
                               'year' and 'block' as the index.
        """
        
        df = df_read_from_sqlite('weather_blocks')
        self.df = df.set_index(['year', 'block'])

    def check_historical_observability(self, time, year=2015, nobs_min=5):
        """Given a (possibly future) UTC time, look up whether PTF 
        was observing at that time in specified year.

        Parameters
        ----------
        time : scalar astropy Time object
            UTC Time
        year : int [2009 -- 2015]
            year to check PTF historical observing
        nobs_min : int (default = 3)
            minimum number of observations per block to count as observable

        Returns
        ----------
        boolean if nobs > nobs_min
        """

        assert((year >= 2009) and (year <= 2015))
        assert((nobs_min > 0))

        block = block_index(time, time_block_size=TIME_BLOCK_SIZE)

        try:
            return self.df.loc[(year, block[0])].values[0] >= nobs_min
        except KeyError:
            # blocks are unfilled if there are no observations
            return False
