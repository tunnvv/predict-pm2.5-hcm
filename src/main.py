from .utils import *
from .model import *

from . import config as c

def main():
    # get global path to these folders `predict-pm2.5-hcm`
    parent_dir = c.parent_dir
    create_folder_for_output(parent_dir)


if __name__ == '__main__':
    main()
