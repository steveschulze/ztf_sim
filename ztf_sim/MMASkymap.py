"""MMA Skymaps."""

import logging
import types

import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
from astropy.time import Time

from .configuration import QueueConfiguration
from .constants import BASE_DIR
from .utils import approx_hours_of_darkness
from .Fields import Fields
from .QueueManager import GreedyQueueManager, RequestPool


class MMASkymap(object):
    """
    A class to handle operations related to Multi-Messenger Astrophysics (MMA) skymaps.
    Attributes:
    -----------
    trigger_name : str
        The name of the trigger event.
    trigger_time : float
        The time of the trigger event.
    skymap_fields : pandas.DataFrame
        A DataFrame containing the skymap fields with 'field_id' and 'probability' columns.
    fields : Fields, optional
        An instance of the Fields class. If not provided, a new instance is created.
    Methods:
    --------
    make_queue(validity_window, observing_fraction=0.5):
        Creates an observation queue based on the skymap fields and the specified observing fraction.
    return_skymap():
        Returns the skymap fields DataFrame.
    persist_skymap():
        Placeholder method to persist the skymap fields.
    archive_persisted_skymap():
        Placeholder method to archive the persisted skymap fields.
    """

    def __init__(self, trigger_name, trigger_time, skymap_fields, fields=None):
        """
        Initialize the MMASkymap object.

        Parameters:
        trigger_name (str): The name of the trigger event.
        trigger_time (datetime): The time of the trigger event.
        skymap_fields (list or dict): The skymap fields data, which should contain 'field_id' and 'probability' columns.
        fields (Fields, optional): An instance of the Fields class. If None, a new Fields instance will be created.

        Raises:
        AssertionError: If 'field_id' or 'probability' are not in skymap_fields.
        """

        self.logger = logging.getLogger(__name__)

        self.trigger_name = trigger_name
        self.trigger_time = trigger_time
        self.skymap_fields = pd.DataFrame(skymap_fields)
        assert('field_id' in self.skymap_fields)
        assert('probability' in self.skymap_fields)

        if fields is None:
            self.fields = Fields()
        else:
            self.fields = fields

    def make_queue(self, validity_window, observing_fraction=0.5):
        """
        Create an observation queue based on the provided validity window and observing fraction.
        Parameters:
        validity_window (list): A list containing the start and end times of the validity window in Modified Julian Date (MJD) format.
        observing_fraction (float, optional): The fraction of the night that should be allocated for observing. Must be between 0 and 1. Default is 0.5.
        Returns:
        queue: An instance of GreedyQueueManager configured with the observation requests.
        Raises:
        AssertionError: If observing_fraction is not between 0 and 1.
        Notes:
        - The function uses a generic configuration and overrides specific parameters.
        - It computes the observability of fields and selects those that are observable within the validity window.
        - The fields are sorted by probability and limited to the number of fields allowed during the night based on the observing fraction.
        - A RequestPool is created with observation requests for the selected fields.
        - A GreedyQueueManager instance is created and returned.
        """

        assert (0 <= observing_fraction <= 1)

        # use a generic configuration and override
        queue_config = QueueConfiguration(BASE_DIR+'../sims/missed_obs.json')
        queue_name = self.trigger_name+'_greedy'
        queue_config.config['queue_name'] = queue_name
        queue_config.config['queue_description'] = queue_name
        queue_config.config['queue_manager'] = 'greedy'
        queue_config.config['observing_programs'] = []
        queue_config.config['validity_window_mjd'] = validity_window

        Time_validity_start = Time(validity_window[0], format='mjd')

        # visibility check
        self.fields.compute_observability(Time_validity_start)
        observable_field_ids = self.fields.select_field_ids(dec_range=[-32,90.],
                           grid_id=0,
                           # use a minimal observable hours cut
                           observable_hours_range=[0.5, 24.])

        # only select fields that are observable tonight and in the primary grid
        w = self.skymap_fields['field_id'].apply(lambda x: x in observable_field_ids)
        skymap_fields = self.skymap_fields.loc[w,:]

        # sort by probability 
        skymap_field_ids = skymap_fields.sort_values(by='probability', ascending=False)['field_id'].values.tolist()
        
        # limit to # of fields allowed during the night
        # for now we're not going to try to propagate in the exact allocation 
        # of observable time; instead we'll just apply a fraction
        # Let's not assume that the validity range provided is only dark time
        dark_time = approx_hours_of_darkness(Time_validity_start,
                                             twilight=coord.Angle(18*u.degree))

        n_fields = int((dark_time * observing_fraction 
                        / (40*u.second) / 2).to(u.dimensionless_unscaled))

        skymap_field_ids = skymap_field_ids[:n_fields]

        w = skymap_fields['field_id'].apply(lambda x: x in skymap_field_ids)
        skymap_fields = skymap_fields.loc[w,:]


        rp = RequestPool()
        for idx, row in skymap_fields.iterrows():
            rp.add_request_sets(1,
                                'MSIP_EMGW',
                                'Kulkarni',
                                int(row['field_id']),
                                [1,2],
                                30*u.minute,
                                30*u.second,
                                2,
                                probability=row['probability'])

        queue = GreedyQueueManager(queue_name, queue_config, rp = rp)

        self.logger.info(f"""Making queue for {self.trigger_name} with """
                         f"""{[(int(row['field_id']), row['probability']) for idx, row in skymap_fields.iterrows()]}""")

        return queue


    def return_skymap(self):
        """
        Returns the skymap fields.

        Returns:
            dict: A dictionary containing the skymap fields.
        """
        
        return self.skymap_fields

    def persist_skymap(self):
        """
        Persist the current state of the skymap.

        This method is intended to save the current skymap to a persistent storage
        medium. The specific implementation details should be provided in the
        method body.

        Note:
            This method currently does not have an implementation.
        """

        pass
    
    def archive_persisted_skymap(self):
        """
        Archives the persisted sky map.

        This method is intended to handle the archiving process for a sky map
        that has been previously persisted. The specific implementation details
        should be defined within this method.

        Note:
            This method currently does not have an implementation.
        """

        pass

