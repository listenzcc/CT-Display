'''
FileName: config.py
Author: Chuncheng
Version: V0.0
Purpose:
'''

# %%
import os

# %%
CONFIG = dict(
    app_name='CT Image Displayer',
    CT_raw_data_folder=os.path.join(os.path.dirname(__file__), '..', 'CT-data')
)
