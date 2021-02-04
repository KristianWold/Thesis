import pickle
import os


PROJECT_ROOT_DIR = "../../results"
DATA_ID = "../../results/data"
FIGURE_ID = "../../results/figures"

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


def saver(object, filename):
    pickle.dump(object, open(filename, "wb"))


def loader(filename):
    object = pickle.load(open(filename, "rb"))

    return object


def unpack_list(list_):
    list_flat = []
    for l in list_:
        list_flat.append(l.flatten())

    list_flat = np.concatenate(list_flat).reshape(-1, 1)

    return list_flat
