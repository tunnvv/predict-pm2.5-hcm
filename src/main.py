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


if __name__ == '__main__':
    main()
