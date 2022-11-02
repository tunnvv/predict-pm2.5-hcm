from .utils import *
from .model import *

from . import config as c

def main():
    # get global path to these folders `predict-pm2.5-hcm`
    parent_dir = c.parent_dir
    create_folder_for_output(parent_dir)

    # read training data to dataframe
    data_dir = os.path.join(parent_dir, 'data', 'training_data', c.fname_data)
    df = load_data(data_dir=data_dir)
    print(df.columns)

    """ ================ PREPARE DATA TRAINING ================ """
    X, y = generate_time_series_data(df, c.window_size, c.stride_pred)
    X = normalize_data(X)
    print("Checking correctness of sample generation ::")
    print(f"The shape of a sample is {X[0].shape}, the correct shape is ({c.window_size}, {c.num_feature}))")


if __name__ == '__main__':
    main()
