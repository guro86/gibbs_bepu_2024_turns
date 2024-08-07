#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:31:01 2024

@author: gustav
"""

import pandas as pd
import numpy as np

rng = np.random.default_rng(
    1986)

dakota_file_path = 'data/fgr/dakota_tabular.dat_old'
meas_file_path = 'data/fgr/measurements.csv_old'

dakota_data = pd.read_csv(
        dakota_file_path,
        sep = '\s+'
    )

#Read measurement data 
meas = pd.read_csv(
    meas_file_path,
    index_col=0
)


meas['campaign'] = 'UNKNOWN'

#Nexps
nexp = 31


order = np.arange(dakota_data.shape[-1])

new_order = np.append(
    order[:-nexp],
    rng.choice(order[-nexp:],replace=False,size=nexp)
    )

#Save data in new order
dakota_data = dakota_data.iloc[:,new_order]

#Save meas in same new order
meas.loc[dakota_data.columns[-nexp:]]

#Create a mapping for renaming the shit
mapping = {name: f'EXP_{i+1}' for i,name in enumerate(meas.index)}

#Make renaming
meas.rename(mapping,axis=0,inplace=True)
dakota_data.rename(mapping,axis=1,inplace=True)

dakota_file_path_new = 'data/fgr/dakota_tabular.dat'
meas_file_path_new = 'data/fgr/measurements.csv'

dakota_data.to_csv(dakota_file_path_new,index=False,sep=' ')
meas.to_csv(meas_file_path_new)