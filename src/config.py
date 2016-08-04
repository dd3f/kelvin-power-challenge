"""Configuration of directories, target column names, cross-validation folds"""

import os
import numpy as np
from datetime import datetime


class ParamConfig:
    def __init__(self):
        #self.target_cols = ['NPWD2372', 'NPWD2401', 'NPWD2402', 'NPWD2451', 'NPWD2471', 'NPWD2472', 'NPWD2481',
        #                    'NPWD2482', 'NPWD2491', 'NPWD2501', 'NPWD2531', 'NPWD2532', 'NPWD2551', 'NPWD2552',
        #                    'NPWD2561', 'NPWD2562', 'NPWD2691', 'NPWD2692', 'NPWD2721', 'NPWD2722',
        #                    'NPWD2742', 'NPWD2771', 'NPWD2791', 'NPWD2792', 'NPWD2801', 'NPWD2802', 'NPWD2821',
        #                    'NPWD2851', 'NPWD2852', 'NPWD2871', 'NPWD2872', 'NPWD2881', 'NPWD2882']
        self.target_cols =['NPWD2531',
                             'NPWD2481',
                             'NPWD2742',
                             'NPWD2491',
                             'NPWD2562',
                             'NPWD2561',
                             'NPWD2721',
                             'NPWD2372',
                             'NPWD2551',
                             'NPWD2771',
                             'NPWD2451',
                             'NPWD2851',
                             'NPWD2532',
                             'NPWD2802',
                             'NPWD2791',
                             'NPWD2881']
        ## path
        self.data_folder = "../../data"
        self.features_folder = "%s/features" % self.data_folder
        self.featuresets_folder = "%s/featuresets" % self.data_folder

        self.models_folder = "%s/models" % self.data_folder
        self.level1_models_folder = "%s/level1" % self.models_folder
        self.model_config_folder = "../../model_config"

        self.log_folder = "../../log/"

        self.folds = [(datetime(2011, 1, 1), datetime(2011, 12, 31)),
                      (datetime(2012, 1, 1), datetime(2012, 12, 31)),
                      (datetime(2013, 1, 1), datetime(2013, 12, 31))]
        
        self.col_to_exclude = ['NPWD2691',
                                 'NPWD2871',
                                 'NPWD2692',
                                 'NPWD2872',
                                 'NPWD2471',
                                 'NPWD2801',
                                 'NPWD2882',
                                 'NPWD2552',
                                 'NPWD2821',
                                 'NPWD2852',
                                 'NPWD2792',
                                 'NPWD2472',
                                 'NPWD2401',
                                 'NPWD2402',
                                 'NPWD2482',
                                 'NPWD2501',
                                 'NPWD2722']


## initialize a param config
config = ParamConfig()
