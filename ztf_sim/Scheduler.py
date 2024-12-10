"""Core scheduler classes."""

import configparser
from collections import defaultdict
import logging
import numpy as np
from astropy.time import Time
import astropy.units as u
from .QueueManager import ListQueueManager, GreedyQueueManager, GurobiQueueManager
from .ObsLogger import ObsLogger
from .configuration import SchedulerConfiguration
from .constants import BASE_DIR, PROGRAM_IDS, EXPOSURE_TIME, READOUT_TIME
from .utils import block_index, block_use_fraction
from .utils import next_12deg_evening_twilight, next_12deg_morning_twilight





class Scheduler(object):
    """
    A class to manage scheduling of observations.
    
    Attributes:
    -----------
    logger : logging.Logger
        Logger for the Scheduler class.
    scheduler_config : SchedulerConfiguration
        Configuration for the scheduler.
    queue_configs : dict
        Configuration for the queues.
    queues : dict
        Dictionary of queues.
    timed_queues_tonight : list
        List of timed queues for tonight.
    mjd_today : int
        Modified Julian Date for today.
    skymaps : dict
        Dictionary of skymaps.
    Q : Queue
        Current active queue.
    run_config : configparser.ConfigParser
        Configuration for the run.
    obs_log : ObsLogger
        Logger for observations.
    
    Methods:
    --------
    __init__(scheduler_config_file_fullpath, run_config_file_fullpath, other_queue_configs=None, output_path=BASE_DIR+'../sims/'):
        Initializes the Scheduler with configuration files and optional queue configurations.
    assign_nightly_requests(current_state_dict, time_limit=15.*u.minute):
        Assigns nightly requests based on the current state and time limit.
    set_queue(queue_name):
        Sets the active queue to the specified queue name.
    add_queue(queue_name, queue, clobber=True):
        Adds a new queue to the scheduler.
    delete_queue(queue_name):
        Deletes a queue from the scheduler.
    add_skymap(trigger_name, skymap, clobber=True):
        Adds a skymap to the scheduler.
    delete_skymap(trigger_name):
        Deletes a skymap from the scheduler.
    find_block_use_tonight(time_now):
        Finds block use for tonight and sets up timed queues for tonight.
    count_timed_observations_tonight():
        Counts the number of timed observations for tonight.
    check_for_TOO_queue_and_switch(time_now):
        Checks if a TOO (Target of Opportunity) queue is now valid and switches to it if necessary.
    check_for_timed_queue_and_switch(time_now):
        Checks if a timed queue is now valid and switches to it if necessary.
    remove_empty_and_expired_queues(time_now):
        Removes empty and expired queues from the scheduler.
    """

    def __init__(self, scheduler_config_file_fullpath, 
            run_config_file_fullpath, other_queue_configs = None,
            output_path = BASE_DIR+'../sims/'):
        """
        Initializes the Scheduler object with the provided configuration files and optional parameters.

        Args:
            scheduler_config_file_fullpath (str): Full path to the scheduler configuration file.
            run_config_file_fullpath (str): Full path to the run configuration file.
            other_queue_configs (optional): Additional queue configurations. Defaults to None.
            output_path (str, optional): Path to the output directory. Defaults to BASE_DIR+'../sims/'.

        Attributes:
            logger (logging.Logger): Logger instance for the scheduler.
            scheduler_config (SchedulerConfiguration): Scheduler configuration object.
            queue_configs (list): List of queue configurations.
            queues (list): List of queues built from the queue configurations.
            timed_queues_tonight (list): List of timed queues for the night.
            mjd_today (int): Modified Julian Date for today, used to trigger nightly recomputes.
            skymaps (dict): Dictionary to store skymaps.
            run_config (configparser.ConfigParser): Configuration parser for the run configuration file.
            obs_log (ObsLogger): Logger for observation history, initialized with the log name and output path.
        """

        self.logger = logging.getLogger(__name__)

        self.scheduler_config = SchedulerConfiguration(
            scheduler_config_file_fullpath)
        self.queue_configs = self.scheduler_config.build_queue_configs()
        self.queues = self.scheduler_config.build_queues(self.queue_configs)
        self.timed_queues_tonight = []

        # used to trigger nightly recomputes
        self.mjd_today = 0

        self.skymaps = {}

        self.set_queue('default')
        
        self.run_config = configparser.ConfigParser()
        self.run_config.read(run_config_file_fullpath)

        if 'log_name' in self.run_config['scheduler']:
            log_name = self.run_config['scheduler']['log_name']
        else:
            log_name = self.scheduler_config.config['run_name']

        # initialize sqlite history
        self.obs_log = ObsLogger(log_name,
                output_path = output_path,
                clobber=self.run_config['scheduler'].getboolean('clobber_db'),) 

    def assign_nightly_requests(self, current_state_dict, 
                                time_limit=15.*u.minute):
        # Look for timed queues that will be valid tonight,
        # to exclude from the nightly solution
        block_use = self.find_block_use_tonight(current_state_dict['current_time'])
        timed_obs_count = self.count_timed_observations_tonight()

        self.logger.info(f'Block use by timed queues: {block_use}')

        self.queues['default'].assign_nightly_requests(
                        current_state_dict,
                        self.obs_log, 
                        time_limit = time_limit,
                        block_use = block_use,
                        timed_obs_count = timed_obs_count,
                        skymaps = self.skymaps)

    def set_queue(self, queue_name): 
        """
        Set the current queue to the specified queue name.

        Parameters:
        queue_name (str): The name of the queue to set as the current queue.

        Raises:
        ValueError: If the specified queue name is not available in the queues.
        """

        if queue_name not in self.queues:
            raise ValueError(f'Requested queue {queue_name} not available!')

        self.Q = self.queues[queue_name]
        
    def add_queue(self,  queue_name, queue, clobber=True):
        """
        Add a queue to the scheduler.

        Parameters:
        queue_name (str): The name of the queue to add.
        queue (object): The queue object to add.
        clobber (bool): If True, overwrite the existing queue with the same name. Default is True.

        Raises:
        ValueError: If clobber is False and a queue with the same name already exists.
        """

        if clobber or (queue_name not in self.queues):
            self.queues[queue_name] = queue 
        else:
            raise ValueError(f"Queue {queue_name} already exists!")

    def delete_queue(self, queue_name):
        """
        Deletes a queue with the given name from the scheduler.

        Args:
            queue_name (str): The name of the queue to delete.

        Raises:
            ValueError: If the queue with the given name does not exist.

        Notes:
            If the queue to be deleted is the current queue, the current queue is set to 'default'.
        """

        if (queue_name in self.queues):
            if self.Q.queue_name == queue_name:
                self.set_queue('default')
            del self.queues[queue_name] 
        else:
            raise ValueError(f"Queue {queue_name} does not exist!")

    def add_skymap(self, trigger_name, skymap, clobber=True):
        """
        Adds a skymap to the scheduler.

        Parameters:
        trigger_name (str): The name of the trigger associated with the skymap.
        skymap (object): The skymap to be added.
        clobber (bool): If True, overwrite any existing skymap with the same trigger_name. Default is True.

        Raises:
        ValueError: If clobber is False and a skymap with the same trigger_name already exists.
        """

        if clobber or (trigger_name not in self.skymaps):
            self.skymaps[trigger_name] = skymap
        else:
            raise ValueError(f"Skymap {trigger_name} already exists!")

    def delete_skymap(self, trigger_name):

        if (trigger_name in self.skymaps):
            del self.skymaps[trigger_name] 
        else:
            raise ValueError(f"Skymap {trigger_name} does not exist!")

    def find_block_use_tonight(self, time_now):
        """
        Determine the block usage for tonight and identify timed queues that will be valid tonight.

        This method calculates the fraction of time blocks that are used during the night, 
        taking into account the evening and morning twilight periods. It also identifies 
        the queues that are valid for tonight and updates the `timed_queues_tonight` attribute.

        Parameters:
        -----------
        time_now : astropy.time.Time
            The current time.

        Returns:
        --------
        block_use : defaultdict
            A dictionary where keys are block indices and values are the fraction of the block 
            that is used tonight.
        """

        # also sets up timed_queues_tonight

        # start of the night
        mjd_today = np.floor(time_now.mjd).astype(int)

        # Look for timed queues that will be valid tonight,
        # to exclude from the nightly solution
        self.timed_queues_tonight = []
        today = Time(mjd_today, format='mjd')
        tomorrow = Time(mjd_today + 1, format='mjd')
        block_start = block_index(today)[0]
        block_stop = block_index(tomorrow)[0]

        block_use = defaultdict(float)

        # compute fraction of twilight blocks not available
        evening_twilight = next_12deg_evening_twilight(today)
        morning_twilight = next_12deg_morning_twilight(today)

        evening_twilight_block = block_index(evening_twilight)[0]
        frac_evening_twilight = block_use_fraction(
                evening_twilight_block, today, evening_twilight)
        block_use[evening_twilight_block] = frac_evening_twilight
        self.logger.debug(f'{frac_evening_twilight} of block {evening_twilight_block} is before 12 degree twilight')

        morning_twilight_block = block_index(morning_twilight)[0]
        frac_morning_twilight = block_use_fraction(
                morning_twilight_block, morning_twilight, tomorrow)
        block_use[morning_twilight_block] = frac_morning_twilight
        self.logger.debug(f'{frac_morning_twilight} of block {morning_twilight_block} is before 12 degree twilight')

        for qq_name, qq in self.queues.items():
            if qq.queue_name in ['default', 'fallback']:
                continue
            if qq.validity_window is not None:
                qq_block_use = qq.compute_block_use()

                is_tonight = False

                # sum block use
                for block, frac in qq_block_use.items():
                    if (block_start <= block <= block_stop):
                        if frac > 0:
                            is_tonight = True
                        self.logger.debug(f'{frac} of block {block} used by queue {qq.queue_name}')
                        block_use[block] += frac
                        if block_use[block] > 1:
                            self.logger.warn(f'Too many observations for block {block}: {block_use[block]}')
                            block_use[block] = 1.

                if is_tonight:    
                    self.timed_queues_tonight.append(qq_name)

        return block_use

    def count_timed_observations_tonight(self):
        """
        Count the number of equivalent observations in timed queues for tonight.
        This method calculates the number of equivalent observations for each program
        in the timed queues scheduled for tonight. It sums up the total observation 
        time for each program and converts it into the number of equivalent observations 
        based on the standard exposure and readout times.

        Returns:
            dict: A dictionary where the keys are program IDs and the values are the 
                  number of equivalent observations for each program.
        """
        # determine how many equivalent obs are in timed queues
        
        timed_obs = {prog:0 for prog in PROGRAM_IDS} 
        if len(self.timed_queues_tonight) == 0:
            return timed_obs

        for qq in self.timed_queues_tonight:
            queue = self.queues[qq].queue.copy()
            if 'n_repeats' not in queue.columns:
                queue['n_repeats'] = 1.
            queue['total_time'] = (queue['exposure_time'] + 
                READOUT_TIME.to(u.second).value)*queue['n_repeats']
            net = queue[['program_id','total_time']].groupby('program_id').agg(np.sum)
            count_equivalent = np.round(net['total_time']/(EXPOSURE_TIME + READOUT_TIME).to(u.second).value).astype(int).to_dict()
            for k, v in count_equivalent.items():
                timed_obs[k] += v

        return timed_obs

    def check_for_TOO_queue_and_switch(self, time_now):
        """
        Check for a Target of Opportunity (TOO) queue and switch to it if necessary.

        This method iterates through all queues to determine if any TOO queue is 
        currently valid based on the provided time. If a valid TOO queue is found, 
        the method will switch to it under the following conditions:
        1. The current queue is not a TOO queue and the TOO queue is not empty.
        2. The current queue is a TOO queue but it is empty, and the new TOO queue 
           is not empty.

        Args:
            time_now (datetime): The current time to check the validity of the TOO queues.
        """

        # check if a TOO queue is now valid
        for qq_name, qq in self.queues.items():
            if qq.is_TOO:
                if qq.is_valid(time_now):
                    # switch if the current queue is not a TOO
                    if (not self.Q.is_TOO) and len(qq.queue):
                        self.set_queue(qq_name)
                    # or if the current TOO queue is empty
                    if ((self.Q.is_TOO) and (len(self.Q.queue) == 0) 
                            and len(qq.queue)):
                        self.set_queue(qq_name)

    def check_for_timed_queue_and_switch(self, time_now):
        """
        Check the current queue and switch to a different queue if necessary based on the current time.

        This method performs the following actions:
        1. If the current queue is not 'default' and is no longer valid, switch to the 'default' queue.
        2. If the current queue is 'default' or 'fallback', check if any other timed queue is now valid.
           - For 'list' type queues, switch to the queue if it has items.
           - For non-'list' type queues, switch to the queue if it has requests within the validity window.

        Args:
            time_now (datetime): The current time used to check the validity of queues.
        """
        
        # drop out of a timed queue if it's no longer valid
        if self.Q.queue_name != 'default':
            if not self.Q.is_valid(time_now):
                self.set_queue('default')

        # only switch from default or fallback queues
        if self.Q.queue_name in ['default', 'fallback']:
            # check if a timed queue is now valid
            for qq_name, qq in self.queues.items():
                if (qq.validity_window is not None) and (qq.is_valid(time_now)): 
                    if (qq.queue_type == 'list'): 
                        # list queues should have items in them
                        if len(qq.queue):
                            self.set_queue(qq_name)
                    else:
                        # don't have a good way to check length of non-list
                        # queues before nightly assignments
                        if qq.requests_in_window:
                            self.set_queue(qq_name)

    def remove_empty_and_expired_queues(self, time_now):
        """
        Remove empty and expired queues from the scheduler.

        This method iterates through the queues in the scheduler and removes those
        that are either empty or have expired based on the current time.

        Args:
            time_now (float): The current time used to check for expired queues.

        Notes:
            - Queues named 'default' and 'fallback' are never removed.
            - A queue is considered expired if its validity window's end time is less than `time_now`.
            - A queue is considered empty if its type is 'list' and it has no elements.
        """
        
        queues_for_deletion = []
        for qq_name, qq in self.queues.items():
            if qq.queue_name in ['default', 'fallback']:
                continue
            if qq.validity_window is not None:
                if qq.validity_window[1] < time_now:
                    self.logger.info(f'Deleting expired queue {qq_name}')
                    queues_for_deletion.append(qq_name)
                    continue
            if (qq.queue_type == 'list') and (len(qq.queue) == 0):
                    self.logger.info(f'Deleting empty queue {qq_name}')
                    queues_for_deletion.append(qq_name)

        # ensure we don't have duplicate values
        queues_for_deletion = set(queues_for_deletion)

        for qq_name in queues_for_deletion:
            self.delete_queue(qq_name)
