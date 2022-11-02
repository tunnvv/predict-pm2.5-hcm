import os

parent_dir = os.getcwd()

config_dir = os.path.join(parent_dir, 'output', 'config')
lgbm_output = os.path.join(parent_dir, 'output', 'lgbm_model')
lstm_output = os.path.join(parent_dir, 'output', 'lstm_model')
img_output = os.path.join(parent_dir, 'output', 'img')