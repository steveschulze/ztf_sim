"""Code for logging observations to a sqlite database."""

import os.path
from collections import defaultdict
import uuid
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import astroplan.moon
from .Fields import Fields
from .utils import *
from .constants import BASE_DIR, FILTER_ID_TO_NAME, EXPOSURE_TIME, READOUT_TIME


class ObsLogger(object):
    """
    A class to log observations for a survey.
    Attributes:
    -----------
    log_name : str
        The name of the log.
    survey_start_time : astropy.time.Time
        The start time of the survey.
    prev_obs : dict or None
        The previous observation record.
    mjd_tonight : float or None
        The Modified Julian Date for tonight.
    moon_illumination_tonight : float or None
        The moon illumination for tonight.
    engine : sqlalchemy.engine.Engine
        The SQLAlchemy engine for the SQLite database.
    conn : sqlalchemy.engine.Connection
        The SQLAlchemy connection to the SQLite database.
    history : pandas.DataFrame
        The DataFrame containing the observation history.
    Methods:
    --------
    __init__(log_name, survey_start_time=Time('2018-01-01'), output_path=BASE_DIR+'../sims/', clobber=False):
        Initializes the ObsLogger with the given parameters.
    create_fields_table(clobber=True):
        Creates the 'Field' table in the database.
    create_pointing_log(clobber=True):
        Creates the 'Summary' table in the database.
    log_pointing(state, request):
        Logs a pointing observation to the database.
    _mjd_filter_history(mjd_range):
        Filters the observation history DataFrame by a given MJD range.
    _equivalent_obs(grp):
        Converts a group of observations to equivalent standard observations.
    count_equivalent_obs_by_program(mjd_range=None):
        Counts the number of equivalent standard exposures by program.
    count_equivalent_obs_by_subprogram(mjd_range=None):
        Counts the number of equivalent standard exposures by program and subprogram.
    count_equivalent_obs_by_program_night(mjd_range=None):
        Counts the number of equivalent standard exposures by program and night.
    select_last_observed_time_by_field(field_ids=None, filter_ids=None, program_ids=None, subprogram_names=None, mjd_range=None):
        Selects the last observed time by field based on given constraints.
    select_n_obs_by_field(field_ids=None, filter_ids=None, program_ids=None, subprogram_names=None, mjd_range=None):
        Selects the number of observations by field based on given constraints.
    return_obs_history(time):
        Returns one night's observation history.
    """

    def __init__(self, log_name, survey_start_time = Time('2018-01-01'),
            output_path = BASE_DIR+'../sims/',
            clobber = False):
        """
        Initialize the ObsLogger object.

        Parameters:
        log_name (str): The name of the log.
        survey_start_time (astropy.time.Time, optional): The start time of the survey. Defaults to Time('2018-01-01').
        output_path (str, optional): The path to the output directory. Defaults to BASE_DIR+'../sims/'.
        clobber (bool, optional): If True, existing tables will be overwritten. Defaults to False.

        Attributes:
        log_name (str): The name of the log.
        survey_start_time (astropy.time.Time): The start time of the survey.
        prev_obs (None): Placeholder for previous observation.
        mjd_tonight (None): Placeholder for Modified Julian Date of tonight.
        moon_illumination_tonight (None): Placeholder for moon illumination of tonight.
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine connected to the SQLite database.
        conn (sqlalchemy.engine.Connection): The connection to the SQLite database.
        history (pandas.DataFrame): The DataFrame containing the summary of the observations.
        """

        self.log_name = log_name
        self.survey_start_time = survey_start_time
        self.prev_obs = None
        self.mjd_tonight = None
        self.moon_illumination_tonight = None
        self.engine = create_engine(
                'sqlite:///'+os.path.join(output_path,f'{self.log_name}.db'))
        self.conn = self.engine.connect()
        self.create_fields_table(clobber=clobber)
        self.create_pointing_log(clobber=clobber)

        self.history = pd.read_sql('Summary', self.engine)

    def create_fields_table(self, clobber=True):
        """
        Creates the 'Field' table in the database and populates it with data.

        Parameters:
        clobber (bool): If True, drops the existing 'Field' table before creating a new one. Default is True.

        The table schema includes the following columns:
        - fieldID: INTEGER PRIMARY KEY
        - fieldFov: REAL
        - fieldRA: REAL
        - fieldDec: REAL
        - fieldGL: REAL
        - fieldGB: REAL
        - fieldEL: REAL
        - fieldEB: REAL

        The method performs the following steps:
        1. Drops the 'Field' table if it exists and clobber is True.
        2. Creates the 'Field' table if it does not exist.
        3. Populates the table with data from the Fields class, renaming columns and setting default values as necessary.
        """

        if clobber:
            # Drop table if it exists
            try:
                self.conn.execute("""DROP TABLE Field""")
            except:
                pass

        # If the table doesn't exist, create it
        if not inspect(self.engine).has_table('Field'): 

            self.conn.execute(text("""
            CREATE TABLE Field(
            fieldID   INTEGER PRIMARY KEY,
            fieldFov  REAL,
            fieldRA   REAL,
            fieldDec  REAL,
            fieldGL   REAL,
            fieldGB   REAL,
            fieldEL   REAL,
            fieldEB   REAL
            )"""))

            f = Fields()
            df = f.fields.reset_index()
            df.rename(columns={'field_id': 'fieldID',
                               'ra': 'fieldRA',
                               'dec': 'fieldDec',
                               'l': 'fieldGL',
                               'b': 'fieldGB',
                               'ecliptic_lon': 'fieldEL',
                               'ecliptic_lat': 'fieldEB'}, inplace=True)
            df.set_index(['fieldID'], inplace=True)
            df['fieldFov'] = 10.428

            df_min = df[['fieldFov','fieldRA', 'fieldDec', 'fieldGL', 'fieldGB',
                'fieldEL', 'fieldEB']]

            # (circumscribed) field diameter in degrees
            df_min.to_sql('Field', self.engine, if_exists='replace')

    def create_pointing_log(self, clobber=True):
        """
        Creates a pointing log table in the database.

        Parameters:
        clobber (bool): If True, drops the existing 'Summary' table if it exists before creating a new one. Default is True.

        Description:
        This method creates a table named 'Summary' in the database if it does not already exist. 
        If the 'clobber' parameter is set to True, it will drop the existing 'Summary' table before creating a new one.
        The 'Summary' table contains the following columns:
            - obsHistID: INTEGER PRIMARY KEY
            - requestID: INTEGER
            - propID: INTEGER
            - fieldID: INTEGER
            - fieldRA: REAL
            - fieldDec: REAL
            - filter: TEXT
            - expDate: INTEGER
            - expMJD: REAL
            - night: INTEGER
            - visitTime: REAL
            - visitExpTime: REAL
            - FWHMgeom: REAL
            - FWHMeff: REAL
            - airmass: REAL
            - filtSkyBright: REAL
            - lst: REAL
            - altitude: REAL
            - azimuth: REAL
            - dist2Moon: REAL
            - solarElong: REAL
            - moonRA: REAL
            - moonDec: REAL
            - moonAlt: REAL
            - moonAZ: REAL
            - moonPhase: REAL
            - sunAlt: REAL
            - sunAz: REAL
            - slewDist: REAL
            - slewTime: REAL
            - fiveSigmaDepth: REAL
            - totalRequestsTonight: INTEGER
            - metricValue: REAL
            - subprogram: TEXT
        """

        if clobber:
            # Drop table if it exists
            try:
                self.conn.execute("""DROP TABLE Summary""")
            except:
                pass

        # If the table doesn't exist, create it
        if not inspect(self.engine).has_table('Summary'): 

            # create table
            self.conn.execute(text("""
            CREATE TABLE Summary(
            obsHistID         INTEGER PRIMARY KEY,
            requestID INTEGER,
            propID INTEGER,
            fieldID      INTEGER,
            fieldRA      REAL,
            fieldDec      REAL,
            filter             TEXT,
            expDate            INTEGER,
            expMJD             REAL,
            night              INTEGER,
            visitTime          REAL,
            visitExpTime       REAL,
            FWHMgeom           REAL,
            FWHMeff            REAL,
            airmass            REAL,
            filtSkyBright      REAL,
            lst                REAL,
            altitude           REAL,
            azimuth            REAL,
            dist2Moon          REAL,
            solarElong         REAL,
            moonRA             REAL,
            moonDec            REAL,
            moonAlt            REAL,
            moonAZ             REAL,
            moonPhase          REAL,
            sunAlt             REAL,
            sunAz              REAL,
            slewDist           REAL,
            slewTime           REAL,
            fiveSigmaDepth     REAL,
            totalRequestsTonight INTEGER,
            metricValue        REAL,
            subprogram         TEXT
            )"""))

    def log_pointing(self, state, request):
        """
        Logs the pointing information of an observation.

        Parameters:
        state (dict): The current state of the observation, including 'current_time' and optionally 'current_zenith_seeing'.
        request (dict): The request details for the observation, including:
            - 'request_id': Unique identifier for the request.
            - 'target_program_id': Identifier for the target program.
            - 'target_field_id': Identifier for the target field.
            - 'target_ra': Right ascension of the target.
            - 'target_dec': Declination of the target.
            - 'target_filter_id': Identifier for the target filter.
            - 'target_exposure_time': Exposure time for the observation.
            - 'target_sky_brightness': Sky brightness for the observation.
            - 'target_limiting_mag': Limiting magnitude for the observation.
            - 'target_total_requests_tonight': Total requests for the night.
            - 'target_metric_value': Metric value for the observation.
            - 'target_subprogram_name': Subprogram name for the observation.

        Returns:
        None
        """

        record = {}
        # don't use request_id here, but
        # let sqlite create a unique non-null key
        #record['obsHistID'] = request['request_id']
        # give request id its own column
        record['requestID'] = request['request_id']
        record['propID'] = request['target_program_id']
        record['fieldID'] = request['target_field_id']
        record['fieldRA'] = np.radians(request['target_ra'])
        record['fieldDec'] = np.radians(request['target_dec'])

        record['filter'] = FILTER_ID_TO_NAME[request['target_filter_id']]
        # times are recorded at start of exposure
        exposure_start = state['current_time'] - \
            request['target_exposure_time']
        # see note in utils.py
        exposure_start.delta_ut1_utc = 0.

        record['expDate'] = (exposure_start - self.survey_start_time).sec
        record['expMJD'] = exposure_start.mjd

        record['night'] = np.floor((exposure_start - self.survey_start_time).jd
                                   ).astype(int)
        record['visitTime'] = request[
            'target_exposure_time'].to(u.second).value
        record['visitExpTime'] = request[
            'target_exposure_time'].to(u.second).value

        # compute some values we will need
        sc = coord.SkyCoord(record['fieldRA'] * u.radian,
                            record['fieldDec'] * u.radian)
        altaz = skycoord_to_altaz(sc, exposure_start)

        if 'current_zenith_seeing' in state:
            pointing_seeing = seeing_at_pointing(state['current_zenith_seeing'].to(
            u.arcsec).value, altaz.alt.value)
            record['FWHMgeom'] = pointing_seeing
            record['FWHMeff'] = pointing_seeing

        record['airmass'] = altaz.secz.value
        record['filtSkyBright'] = request['target_sky_brightness']
        # despite the docs, it seems lst is stored as radians
        record['lst'] = np.radians(exposure_start.sidereal_time('apparent').to(
            u.hourangle).value/24.*360.)
        record['altitude'] = altaz.alt.to(u.radian).value
        record['azimuth'] = altaz.az.to(u.radian).value

        sun = coord.get_sun(exposure_start)
        sun_altaz = skycoord_to_altaz(sun, exposure_start)
        moon = coord.get_moon(exposure_start, P48_loc)
        moon_altaz = skycoord_to_altaz(moon, exposure_start)

        # WORKING AROUND a bug in sc.separation(moon)!
        #moon_sc = coord.SkyCoord(moon.ra,moon.dec)
        record['dist2Moon'] = moon.separation(sc).to(u.radian).value
        record['solarElong'] = sun.separation(sc).to(u.deg).value
        record['moonRA'] = moon.ra.to(u.radian).value
        record['moonDec'] = moon.dec.to(u.radian).value
        record['moonAlt'] = moon_altaz.alt.to(u.radian).value
        record['moonAZ'] = moon_altaz.az.to(u.radian).value

        # store tonight's mjd so that we can avoid recomputing moon
        # illumination, which profiling shows is weirdly expensive
        if np.floor(exposure_start.mjd) != self.mjd_tonight:
            self.moon_illumination_tonight = astroplan.moon.moon_illumination(
                # Don't use P48_loc to avoid astropy bug:
                # https://github.com/astropy/astroplan/pull/213
                # exposure_start, P48_loc) * 100.
                exposure_start) * 100.
            self.mjd_tonight = np.floor(exposure_start.mjd)

        record['moonPhase'] = self.moon_illumination_tonight

        record['sunAlt'] = sun_altaz.alt.to(u.radian).value
        record['sunAz'] = sun_altaz.az.to(u.radian).value
        if self.prev_obs is not None:
            sc_prev = coord.SkyCoord(self.prev_obs['fieldRA'] * u.radian,
                                     self.prev_obs['fieldDec'] * u.radian)
            record['slewDist'] = sc.separation(sc_prev).to(u.radian).value
            record['slewTime'] = (record['expDate'] -
                                  (self.prev_obs['expDate'] +
                                      self.prev_obs['visitTime']))
        record['fiveSigmaDepth'] = request['target_limiting_mag']

        # ztf_sim specific keywords!
        record['totalRequestsTonight'] = \
            request['target_total_requests_tonight']
        record['metricValue'] = request['target_metric_value']
        record['subprogram'] = request['target_subprogram_name'] 

        record_row = pd.DataFrame(record,index=[uuid.uuid1().hex])

        # append to our local history DataFrame
        # note that the index here will change when reloaded from the db
        self.history = pd.concat([self.history, record_row], axis=0, join='outer',
                                           sort=False)

        # write to the database
        record_row.to_sql('Summary', self.conn, index=False, if_exists='append')

#        # convert nan to SQL NULL. might be smarter to just replace the
#        # insertion method below with something smarter (pd.to_sql?)
#        for k,v in record.items():
#            try:
#                if np.isnan(v):
#                    record[k] = 'NULL'
#            except TypeError:
#                continue
#
#        # use placeholders to create the INSERT query
#        columns = ', '.join(list(record.keys()))
#        placeholders = '{' + '}, {'.join(list(record.keys())) + '}'
#        query = 'INSERT INTO Summary ({}) VALUES ({})'.format(
#            columns, placeholders)
#        query_filled = query.format(**record)
#        self.conn.execute(query_filled)


        # save record for next obs
        self.prev_obs = record

    def _mjd_filter_history(self, mjd_range):
        """
        Filters the observation history based on a given Modified Julian Date (MJD) range.

        Parameters:
        mjd_range (tuple or list): A tuple or list containing two elements representing 
                                   the start and end of the MJD range. If `None`, the 
                                   entire history is returned.

        Returns:
        pandas.DataFrame: A dataframe containing the filtered observation history within 
                          the specified MJD range. If `mjd_range` is `None`, the entire 
                          history is returned.

        Raises:
        AssertionError: If the start of the MJD range is greater than the end.
        """

        if mjd_range is not None:
            assert mjd_range[0] <= mjd_range[1]
            w = ((self.history['expMJD'] >= mjd_range[0]) & 
                  (self.history['expMJD'] <= mjd_range[1])) 
            hist = self.history[w]
        else:
            hist = self.history

        return hist

    def _equivalent_obs(self, grp):
        """
        Given a dataframe groupby object, convert to equivalent standard observations.

        Parameters
        ----------
        grp : pandas.core.groupby.DataFrameGroupBy
            A dataframe groupby object containing observation data.

        Returns
        -------
        defaultdict
            A dictionary with keys determined by the group, where values are the equivalent 
            number of standard observations. The dictionary is a defaultdict with a default 
            value of zero for new programs.

        Notes
        -----
        The function calculates the total exposure time and the number of observations, 
        adds readout overhead (excluding slew time), and then computes the equivalent 
        number of standard observations based on the total time divided by the sum of 
        exposure time and readout time.
        """

        total_exposure_time = grp['visitExpTime'].agg(np.sum)
        count_nobs = grp['requestID'].agg(len)

        # add readout overhead (but not slew)
        total_time = total_exposure_time + count_nobs * READOUT_TIME.to(u.second).value
        count_equivalent = np.round(total_time/(EXPOSURE_TIME + READOUT_TIME).to(u.second).value).astype(int).to_dict()

        # make this a defaultdict so we get zero values for new programs
        return defaultdict(int, count_equivalent)


    def count_equivalent_obs_by_program(self, mjd_range = None):
        """
        Count the number of equivalent standard exposures by program.
        Parameters:
        mjd_range (tuple, optional): A tuple containing the start and end Modified Julian Date (MJD) range 
                         to filter the observations. If None, all observations are considered.
        Returns:
        pandas.DataFrame: A DataFrame with two columns:
                  - 'program_id': The ID of the program.
                  - 'n_obs': The number of equivalent standard exposures for each program.
        """

        hist = self._mjd_filter_history(mjd_range)

        grp = hist.groupby(['propID'])

        s = pd.Series(self._equivalent_obs(grp))
        s.index.name = 'program_id'
        s.name = 'n_obs'
        s = s.reset_index()
        return s

    def count_equivalent_obs_by_subprogram(self, mjd_range = None):
        """
        Count the number of equivalent standard exposures by program and subprogram.

        This method filters the observation history based on the provided MJD (Modified Julian Date) range,
        groups the filtered history by 'propID' and 'subprogram', and then calculates the number of equivalent
        standard exposures for each group.

        Parameters:
        mjd_range (tuple, optional): A tuple containing the start and end of the MJD range to filter the history.
                                     If None, the entire history is used.

        Returns:
        pandas.DataFrame: A DataFrame with columns 'program_id', 'subprogram_name', and 'n_obs', where 'n_obs'
                          represents the count of equivalent standard exposures for each program and subprogram.
                          If there is no history, an empty DataFrame with the same columns is returned.
        """

        hist = self._mjd_filter_history(mjd_range)

        grp = hist.groupby(['propID','subprogram'])

        s = pd.Series(self._equivalent_obs(grp))
        if len(s):
            s.index.names = ['program_id','subprogram_name']
            s.name = 'n_obs'
            s = s.reset_index()
        else:
            # handle no history
            s = pd.DataFrame(columns = ['program_id','subprogram_name', 'n_obs'])
        return s

    def count_equivalent_obs_by_program_night(self, mjd_range = None):
        """
        Count the number of equivalent standard exposures by program, subprogram, and night.

        Parameters:
        mjd_range (tuple, optional): A tuple specifying the range of Modified Julian Dates (MJD) to filter the observations. 
                                     If None, all observations are considered.

        Returns:
        pandas.DataFrame: A DataFrame with columns ['program_id', 'night', 'n_obs'] where 'n_obs' is the count of equivalent 
                          standard exposures for each program and night.
        """

        hist = self._mjd_filter_history(mjd_range)

        grp = hist.groupby(['propID','night'])

        s = pd.Series(self._equivalent_obs(grp))
        s.index.names = ['program_id','night']
        s.name = 'n_obs'
        s = s.reset_index()
        return s

    def select_last_observed_time_by_field(self,
            field_ids = None, filter_ids = None, 
            program_ids = None, subprogram_names = None, 
            mjd_range = None):
        """
        Select the last observed time by field based on various criteria.

        Parameters:
        -----------
        field_ids : list, optional
            List of field IDs to filter by.
        filter_ids : list, optional
            List of filter IDs to filter by.
        program_ids : list, optional
            List of program IDs to filter by.
        subprogram_names : list, optional
            List of subprogram names to filter by.
        mjd_range : tuple, optional
            Tuple specifying the range of Modified Julian Dates (MJD) to filter by.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the last observed time for each field that meets the specified criteria.
            The DataFrame has 'fieldID' as the index and 'expMJD' as the column, where 'expMJD' is the 
            maximum observed time for each field.
        """

        # start with "True" 
        w = self.history['expMJD'] > 0

        if field_ids is not None:
            w &= self.history['fieldID'].apply(lambda x: x in field_ids)

        if filter_ids is not None:
            filter_names = [FILTER_ID_TO_NAME[fi] for fi in filter_ids]
            w &= self.history['filter'].apply(lambda x: 
                    x in filter_names)

        if program_ids is not None:
            w &= self.history['propID'].apply(lambda x: 
                    x in program_ids)

        if subprogram_names is not None:
            w &= self.history['subprogram'].apply(lambda x: 
                    x in subprogram_names)

        if mjd_range is not None:
            assert mjd_range[0] <= mjd_range[1]
            w &= ((self.history['expMJD'] >= mjd_range[0]) & 
                  (self.history['expMJD'] <= mjd_range[1])) 

        # note that this only returns fields that have previously 
        # been observed under these constraints!
        return self.history.loc[
                w,['fieldID','expMJD']].groupby('fieldID').agg(np.max)

    def select_n_obs_by_field(self, field_ids=None, filter_ids=None, program_ids=None, subprogram_names=None, mjd_range=None):
        """
        Select the number of observations by field based on various criteria.

        Parameters:
        -----------
        field_ids : list, optional
            List of field IDs to filter the observations.
        filter_ids : list, optional
            List of filter IDs to filter the observations.
        program_ids : list, optional
            List of program IDs to filter the observations.
        subprogram_names : list, optional
            List of subprogram names to filter the observations.
        mjd_range : tuple, optional
            Tuple containing the start and end of the MJD (Modified Julian Date) range to filter the observations.

        Returns:
        --------
        pandas.Series
            A series with field IDs as the index and the number of observations as the values.

        Notes:
        ------
        This method only returns fields that have previously been observed.
        """

        # start with "True" 
        w = self.history['expMJD'] > 0

        if field_ids is not None:
            w &= self.history['fieldID'].apply(lambda x: x in field_ids)

        if filter_ids is not None:
            filter_names = [FILTER_ID_TO_NAME[fi] for fi in filter_ids]
            w &= self.history['filter'].apply(lambda x: 
                    x in filter_names)

        if program_ids is not None:
            w &= self.history['propID'].apply(lambda x: 
                    x in program_ids)

        if subprogram_names is not None:
            w &= self.history['subprogram'].apply(lambda x: 
                    x in subprogram_names)

        if mjd_range is not None:
            assert mjd_range[0] <= mjd_range[1]
            w &= ((self.history['expMJD'] >= mjd_range[0]) & 
                  (self.history['expMJD'] <= mjd_range[1])) 

        # note that this only returns fields that have previously 
        # been observed!   
        grp =  self.history.loc[
                w,['fieldID','expMJD']].groupby('fieldID')
        nobs = grp['expMJD'].agg(len)
        nobs.name = 'n_obs'

        return nobs

    def return_obs_history(self, time):
        """
        Return one night's observation history.

        Parameters:
        time (astropy.time.Time): The time object representing the observation time.

        Returns:
        pandas.DataFrame: A DataFrame containing the observation history for the specified night.
                          The DataFrame includes the following columns:
                          - 'requestID': The request ID of the observation.
                          - 'propID': The proposal ID of the observation.
                          - 'fieldID': The field ID of the observation.
                          - 'fieldRA': The right ascension of the field.
                          - 'fieldDec': The declination of the field.
                          - 'filter': The filter used for the observation.
                          - 'expMJD': The modified Julian date of the observation.
                          - 'visitExpTime': The exposure time of the visit.
                          - 'airmass': The airmass during the observation.
                          - 'subprogram': The subprogram associated with the observation.
        """

        mjd_range = [np.floor(time.mjd), np.floor(time.mjd)+1.]
        w = ((self.history['expMJD'] >= mjd_range[0]) & 
                  (self.history['expMJD'] <= mjd_range[1])) 
        return self.history.loc[w, 
                ['requestID', 'propID', 'fieldID', 
                    'fieldRA', 'fieldDec', 'filter', 'expMJD', 'visitExpTime',
                    'airmass', 'subprogram']]

