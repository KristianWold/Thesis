import pickle
import os


PROJECT_ROOT_DIR = "../results"
DATA_ID = "../results/data"
FIGURE_ID = "../results/figures"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)
if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)
if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)


def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(data_id):
    return os.path.join(DATA_ID, data_id)


def identity(func):
    return func


def loader(filename):
    object = pickle.load(open(filename, "rb"))

    return object
