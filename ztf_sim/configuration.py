"""Classes for parsing scheduler configuration files."""

import pathlib
import json
import numpy as np
import astropy.units as u
from .ObservingProgram import ObservingProgram
from .Fields import Fields
from .constants import PROGRAM_NAMES, PROGRAM_NAME_TO_ID, EXPOSURE_TIME, TIME_BLOCK_SIZE
from .QueueManager import GreedyQueueManager, QueueEmptyError, GurobiQueueManager, ListQueueManager
from .field_selection_functions import *


class Configuration(object):
    """
    A class used to represent the Configuration.

    Attributes
    ----------
    config : dict
        A dictionary to store the configuration settings.

    Methods
    -------
    __init__(self, config_file)
        Initializes the Configuration object and loads the configuration from the given file if provided.
    
    load_configuration(self, config_file)
        Loads the configuration from the given JSON file.
    """

    def __init__(self, config_file):
        """
        Initializes the Configuration object.

        Parameters
        ----------
        config_file : str
            The path to the configuration JSON file.
        """

        """
        Loads the configuration from the given JSON file.

        Parameters
        ----------
        config_file : str
            The path to the configuration JSON file.
        """

        if config_file is not None:
            self.load_configuration(config_file)

    def load_configuration(self, config_file):
        """
        Load configuration from a JSON file.

        Args:
            config_file (str): Path to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the file is not a valid JSON.

        Sets:
            self.config (dict): The loaded configuration.
        """
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.config = config

class SchedulerConfiguration(Configuration):
    """
    SchedulerConfiguration is a class that handles the configuration for a scheduler.
    Attributes:
        scheduler_config_file (pathlib.PurePosixPath): The path to the scheduler configuration file.
    Methods:
        __init__(config_file):
            Initializes the SchedulerConfiguration with the given configuration file.
        check_configuration():
            Checks if the configuration is valid. Raises ValueError if the configuration is invalid.
        build_queue_configs():
            Builds and returns a dictionary of queue configurations.
        build_queues(queue_configs):
            Builds and returns a dictionary of queues based on the provided queue configurations.
    """

    def __init__(self, config_file):
        """
        Initialize the configuration with the given config file.

        Args:
            config_file (str): Path to the configuration file.

        """
        super().__init__(config_file)
        self.scheduler_config_file = pathlib.PurePosixPath(config_file)
        self.check_configuration()

    def check_configuration(self):
        """
        Validates the scheduler configuration.

        This method checks if the configuration contains the required 'queues' key.
        It ensures that each queue in the configuration has a 'queue_name' and a 'config_file'.
        Additionally, it verifies that there is a queue named 'default'.

        Raises:
            ValueError: If 'queues' is not present in the configuration.
            ValueError: If no queue named 'default' is found.
            AssertionError: If any queue does not have 'queue_name' or 'config_file'.
        """
        
        if 'queues' not in self.config:
            raise ValueError("Scheduler configuration must give queues")
        has_default = False
        for queue_pars in self.config['queues']:
            assert "queue_name" in queue_pars
            if queue_pars["queue_name"] == "default":
                has_default = True
            assert "config_file" in queue_pars
        if not has_default:
            raise ValueError("Scheduler configuration must specify a default queue")

    def build_queue_configs(self):
        """
        Builds and returns a dictionary of queue configurations.

        This method reads the queue configurations specified in the 'queues' section
        of the main configuration and creates a QueueConfiguration object for each queue.
        The resulting dictionary maps queue names to their respective QueueConfiguration objects.

        Returns:
            dict: A dictionary where the keys are queue names and the values are QueueConfiguration objects.

        Raises:
            Exception: If there is an error reading any of the queue configuration files.
        """

        queue_configs = {}

        for queue_pars in self.config['queues']:
            try:
                queue_config = QueueConfiguration(
                    self.scheduler_config_file.parent / queue_pars["config_file"])
            except Exception as e:
                print(f'Error reading config file {queue_pars["config_file"]}')
                raise(e)

            queue_configs[queue_pars["queue_name"]] = queue_config

        return queue_configs

    def build_queues(self, queue_configs):
        """
        Build and return a dictionary of queue managers based on the provided configurations.
        Args:
            queue_configs (dict): A dictionary where keys are queue names and values are 
                                  configuration objects containing a 'queue_manager' key 
                                  with one of the following values: 'list', 'greedy', 'gurobi'.
        Returns:
            dict: A dictionary where keys are queue names and values are instances of 
                  ListQueueManager, GreedyQueueManager, or GurobiQueueManager based on 
                  the 'queue_manager' specified in the configuration.
        Raises:
            AssertionError: If the 'queue_manager' value is not one of 'list', 'greedy', or 'gurobi'.
            Exception: If there is an error while creating a queue manager instance.
        """
        
        queues = {}
        for queue_name, queue_config in queue_configs.items():
            
            queue_manager = queue_config.config['queue_manager']
            assert (queue_manager in ('list', 'greedy', 'gurobi'))

            try:
                if queue_manager == 'list':
                    queues[queue_name] = ListQueueManager(queue_name, queue_config)
                elif queue_manager == 'greedy':
                    queues[queue_name] = GreedyQueueManager(queue_name, queue_config)
                elif queue_manager == 'gurobi':
                    queues[queue_name] = GurobiQueueManager(queue_name, queue_config)
            except Exception as e:
                print(f'Error building queue {queue_name}')
                raise(e)


        return queues

class QueueConfiguration(Configuration):
    """
    A class used to represent the configuration for a queue-based observing program.

    Attributes
    ----------
    config_file : str
        The path to the configuration file.

    Methods
    -------
    check_configuration():
        Validates the configuration to ensure observing fractions sum to 1 and program names are valid.
    
    build_observing_programs():
        Constructs observing programs based on the configuration.
    """

    def __init__(self, config_file):
        """
        Initializes the QueueConfiguration with a configuration file and checks the configuration.

        Parameters
        ----------
        config_file : str
            The path to the configuration file.
        """

        super().__init__(config_file)
        self.check_configuration()

    def check_configuration(self):
        """
        Validates the configuration to ensure observing fractions sum to 1 for each month and program names are valid.

        Raises
        ------
        ValueError
            If the observing fractions do not sum to 1 or if a program name is not in the known programs.
        """

        if self.config['queue_manager'] != 'list' and len(self.config['observing_programs']):
            for month in range(1,13):
                if not np.isclose(np.sum(
                    [prog['program_observing_fraction']*prog['subprogram_fraction'] 
                    for prog in self.config['observing_programs']
                    if ((prog['active_months'] == 'all') or 
                    (month in np.atleast_1d(prog['active_months'])))
                    ]), 1.0):
                    raise ValueError(f"Observing fractions must sum to 1: {[(prog['subprogram_name'], prog['program_observing_fraction']*prog['subprogram_fraction']) for prog in self.config['observing_programs']]}")

            # could do this via schema validation
            for prog in self.config['observing_programs']:
                if prog['program_name'] not in PROGRAM_NAMES:
                    raise ValueError('{} not in known programs'.format(
                        prog['program_name']))

    def build_observing_programs(self):
        """
        Constructs observing programs based on the configuration.

        Returns
        -------
        list
            A list of ObservingProgram objects.

        Raises
        ------
        ValueError
            If field_ids are not valid or if field_selection_function does not exist.
        """

        OPs = []
        f = Fields()
        for prog in self.config['observing_programs']:
            assert(('field_ids' in prog) or ('field_selections' in prog)
                    or ('field_selection_function' in prog))
            assert(('field_ids' in prog) + ('field_selections' in prog) + 
                    ('field_selection_function' in prog) == 1)
            if 'field_ids' in prog:
                field_ids = prog['field_ids']
                for field_id in field_ids:
                    if field_id not in f.fields.index:
                        raise ValueError(f'Input field_id {field_id} is not valid')
                field_selection_function = None
            elif 'field_selections' in prog: 
                field_ids = f.select_field_ids(**prog['field_selections'])
                field_selection_function = None
            else:
                field_ids = None
                field_selection_function = prog['field_selection_function']
                # check if it exists
                assert(field_selection_function in globals())
            if 'nobs_range' not in prog:
                prog['nobs_range'] = None
            if 'intranight_gap_min' not in prog:
                prog['intranight_gap_min'] = TIME_BLOCK_SIZE
            else:
                # make it a quantity
                prog['intranight_gap_min'] = prog['intranight_gap_min'] * u.minute 
            if 'exposure_time' not in prog:
                prog['exposure_time'] = EXPOSURE_TIME
            else:
                # make it a quantity
                prog['exposure_time'] = prog['exposure_time'] * u.second
            OP = ObservingProgram(PROGRAM_NAME_TO_ID[prog['program_name']],
                                  prog['subprogram_name'], 
                                  prog['program_pi'], 
                                  prog['program_observing_fraction'],
                                  prog['subprogram_fraction'],
                                  field_ids, prog['filter_ids'],
                                  prog['internight_gap_days'] * u.day,
                                  prog['intranight_gap_min'],
                                  prog['n_visits_per_night'],
                                  exposure_time = prog['exposure_time'],
                                  nobs_range = prog['nobs_range'],
                                  filter_choice=prog['filter_choice'],
                                  active_months=prog['active_months'],
                                  field_selection_function = field_selection_function)
            OPs.append(OP)

        return OPs
