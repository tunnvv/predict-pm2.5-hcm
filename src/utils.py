import os

def create_folder_for_output(parent_dir):
    """ create an output folder,
    to store all the outputs of the trained model:
    weights of models, metrics of models """

    def create_folder(folder_path):
        folder_path = os.path.abspath(folder_path)
        project_folder = os.path.dirname(os.getcwd())

        if os.path.commonpath([folder_path, project_folder]) != project_folder:
            return -1
        if folder_path == project_folder:
            return 0
        if os.path.isdir(folder_path):
            return 0
        create_folder(os.path.dirname(folder_path))
        os.mkdir(folder_path)
        return 0

    create_folder(os.path.join(parent_dir, 'output'))

    create_folder(os.path.join(parent_dir, 'output', 'config'))
    create_folder(os.path.join(parent_dir, 'output', 'img'))
    create_folder(os.path.join(parent_dir, 'output', 'lgbm_model'))
    create_folder(os.path.join(parent_dir, 'output', 'lstm_model'))
    create_folder(os.path.join(parent_dir, 'output', 'results'))

    create_folder(os.path.join(parent_dir, 'output', 'img', 'comp_graph'))
    create_folder(os.path.join(parent_dir, 'output', 'img', 'scatter_plot'))

    create_folder(os.path.join(parent_dir, 'output', 'lstm_model', 'loggers'))