import os

MODELS_DIR = '/path/to/TRACIE_MODELS'
DATA_DIR = '/path/to/tracie'

SYMTIME_PRETRAINED_MODEL_DURATION = os.path.join(MODELS_DIR, 'symtime-pretrained-model/duration')
SYMTIME_PRETRAINED_MODEL_START = os.path.join(MODELS_DIR, 'symtime-pretrained-model/start')
PTNTIME_PRETRAINED_MODEL = os.path.join(MODELS_DIR, 'ptntime-pretrained-model')

IID_TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'data/iid/tracie_train.txt')
IID_EVAL_DATA_FILE = os.path.join(DATA_DIR, 'data/iid/tracie_test.txt')

UNIFORM_TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'data/uniform-prior/tracie_train_uniform_prior.txt')
UNIFORM_EVAL_DATA_FILE = os.path.join(DATA_DIR, 'data/uniform-prior/tracie_test.txt')

IID_TRAIN_DATA_FILE_SYMBOLIC = os.path.join(DATA_DIR, 'data/iid-symbolic-format/train.txt')
IID_EVAL_DATA_FILE_SYMBOLIC = os.path.join(DATA_DIR, 'data/iid-symbolic-format/test.txt')

UNIFORM_TRAIN_DATA_FILE_SYMBOLIC = os.path.join(DATA_DIR, 'data/uniform-prior-symbolic-format/train.txt')
UNIFORM_EVAL_DATA_FILE_SYMBOLIC = os.path.join(DATA_DIR, 'data/uniform-prior-symbolic-format/test.txt')

MATRES_MINIMAL_SUPERVISION_TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'data/matres/matres_minimal_supervision_train.txt')
MATRES_TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'data/matres/matres_train_before_after_tracie_style.txt')
MATRES_EVAL_DATA_FILE = os.path.join(DATA_DIR, 'data/matres/matres_test_before_after_tracie_style.txt')

config_dict = {
    'iid': {
        'base_model': {'model_name_or_path': 't5-large',
                       'output_dir': os.path.join(DATA_DIR, 'experiments/iid/base'),
                       'train_data_file': IID_TRAIN_DATA_FILE, 'eval_data_file': IID_EVAL_DATA_FILE},
        'ptntime': {'model_name_or_path': PTNTIME_PRETRAINED_MODEL,
                    'output_dir': os.path.join(DATA_DIR, 'experiments/iid/ptntime'),
                    'train_data_file': IID_TRAIN_DATA_FILE, 'eval_data_file': IID_EVAL_DATA_FILE},
        'symtime': {'model_name_or_path': SYMTIME_PRETRAINED_MODEL_START,
                    'duration_model_path': SYMTIME_PRETRAINED_MODEL_DURATION,
                    'output_dir': os.path.join(DATA_DIR, 'experiments/iid/symtime'),
                    'train_data_file': IID_TRAIN_DATA_FILE_SYMBOLIC,
                    'eval_data_file': IID_EVAL_DATA_FILE_SYMBOLIC}
    },
    'uniform': {
        'base_model': {'model_name_or_path': 't5-large',
                       'output_dir': os.path.join(DATA_DIR, 'experiments/uniform/base'),
                       'train_data_file': UNIFORM_TRAIN_DATA_FILE, 'eval_data_file': UNIFORM_EVAL_DATA_FILE},
        'ptntime': {'model_name_or_path': PTNTIME_PRETRAINED_MODEL,
                    'output_dir': os.path.join(DATA_DIR, 'experiments/uniform/ptntime'),
                    'train_data_file': UNIFORM_TRAIN_DATA_FILE, 'eval_data_file': UNIFORM_EVAL_DATA_FILE},
        'symtime': {'model_name_or_path': SYMTIME_PRETRAINED_MODEL_START,
                    'duration_model_path': SYMTIME_PRETRAINED_MODEL_DURATION,
                    'output_dir': os.path.join(DATA_DIR, 'experiments/uniform/symtime'),
                    'train_data_file': UNIFORM_TRAIN_DATA_FILE_SYMBOLIC, 'eval_data_file': UNIFORM_EVAL_DATA_FILE_SYMBOLIC}
    },
    'matres': {
        'base_model': {'model_name_or_path': 't5-large',
                       'output_dir': os.path.join(DATA_DIR, 'experiments/matres/base'),
                       'train_data_file': MATRES_MINIMAL_SUPERVISION_TRAIN_DATA_FILE, 'eval_data_file': MATRES_EVAL_DATA_FILE},
        'ptntime': {'model_name_or_path': PTNTIME_PRETRAINED_MODEL,
                    'output_dir': os.path.join(DATA_DIR, 'experiments/matres/ptntime'),
                    'train_data_file': MATRES_MINIMAL_SUPERVISION_TRAIN_DATA_FILE, 'eval_data_file': MATRES_EVAL_DATA_FILE}
    }
}