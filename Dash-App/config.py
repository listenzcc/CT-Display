'''
FileName: config.py
Author: Chuncheng
Version: V0.0
Purpose:
'''

# %%
import os
import logging

# %%
CONFIG = dict(
    app_name='CT Image Displayer',
    CT_raw_data_folder=os.path.join(
        os.path.dirname(__file__), '..', 'CT-data')
)

# %%
logger_kwargs = dict(
    level_file=logging.DEBUG,
    level_console=logging.DEBUG,
    format_file='%(asctime)s %(name)s %(levelname)-8s %(message)-40s {{%(filename)s:%(lineno)s:%(module)s:%(funcName)s}}',
    format_console='%(asctime)s %(name)s %(levelname)-8s %(message)-40s {{%(filename)s:%(lineno)s}}'
)


def generate_logger(name, filepath, level_file, level_console, format_file, format_console):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(logging.Formatter(format_file))
    file_handler.setLevel(level_file)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_console))
    console_handler.setLevel(level_console)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = generate_logger('CTDisplay', 'CTDisplay.log', **logger_kwargs)
logger.info(
    '--------------------------------------------------------------------------------')
logger.info(
    '---- New Session is started ----------------------------------------------------')

# %%
