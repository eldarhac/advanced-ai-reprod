SYMTIME_PRETRAINED_MODEL_DURATION = '/content/drive/MyDrive/TRACIE_MODELS/symtime-pretrained-model/duration'
SYMTIME_PRETRAINED_MODEL_START = '/content/drive/MyDrive/TRACIE_MODELS/symtime-pretrained-model/start'
PTNTIME_PRETRAINED_MODEL = '/content/drive/MyDrive/TRACIE_MODELS/ptntime-pretrained-model'

IID_TRAIN_DATA_FILE = 'data/iid/tracie_train.txt'
IID_EVAL_DATA_FILE = 'data/iid/tracie_test.txt'

UNIFORM_TRAIN_DATA_FILE = 'data/uniform-prior/tracie_train_uniform_prior.txt'
UNIFORM_EVAL_DATA_FILE = 'data/uniform-prior/tracie_test.txt'

IID_TRAIN_DATA_FILE_SYMBOLIC = 'data/iid-symbolic-format/train.txt'
IID_EVAL_DATA_FILE_SYMBOLIC = 'data/iid-symbolic-format/test.txt'

UNIFORM_TRAIN_DATA_FILE_SYMBOLIC = 'data/uniform-prior-symbolic-format/train.txt'
UNIFORM_EVAL_DATA_FILE_SYMBOLIC = 'data/uniform-prior-symbolic-format/test.txt'


MATRES_TRAIN_DATA_FILE = 'data/matres/matres_train_before_after_tracie_style.txt'
MATRES_EVAL_DATA_FILE = 'data/matres/matres_test_before_after_tracie_style.txt'

config_dict = {
    'iid': {
        'base_model': {'model_name_or_path': 't5-large', 'output_dir': 'experiments/iid/base',
                       'train_data_file': IID_TRAIN_DATA_FILE, 'eval_data_file': IID_EVAL_DATA_FILE},
        'ptntime': {'model_name_or_path': PTNTIME_PRETRAINED_MODEL,
                    'output_dir': 'experiments/iid/ptntime',
                    'train_data_file': IID_TRAIN_DATA_FILE, 'eval_data_file': IID_EVAL_DATA_FILE},
        'symtime': {'model_name_or_path': SYMTIME_PRETRAINED_MODEL_START,
                    'duration_model_path': SYMTIME_PRETRAINED_MODEL_DURATION,
                    'output_dir': 'experiments/iid/symtime',
                    'train_data_file': IID_TRAIN_DATA_FILE_SYMBOLIC, 'eval_data_file': IID_EVAL_DATA_FILE_SYMBOLIC}
    },
    'uniform': {
        'base_model': {'model_name_or_path': 't5-large', 'output_dir': 'experiments/uniform/base',
                       'train_data_file': UNIFORM_TRAIN_DATA_FILE, 'eval_data_file': UNIFORM_EVAL_DATA_FILE},
        'ptntime': {'model_name_or_path': PTNTIME_PRETRAINED_MODEL,
                    'output_dir': 'experiments/uniform/ptntime',
                    'train_data_file': UNIFORM_TRAIN_DATA_FILE, 'eval_data_file': UNIFORM_EVAL_DATA_FILE},
        'symtime': {'model_name_or_path': SYMTIME_PRETRAINED_MODEL_START,
                    'duration_model_path': SYMTIME_PRETRAINED_MODEL_DURATION,
                    'output_dir': 'experiments/uniform/symtime',
                    'train_data_file': UNIFORM_TRAIN_DATA_FILE_SYMBOLIC, 'eval_data_file': UNIFORM_EVAL_DATA_FILE_SYMBOLIC}
    },
    'matres': {
        'base_model': {'model_name_or_path': 't5-large', 'output_dir': 'experiments/matres/base',
                       'train_data_file': MATRES_TRAIN_DATA_FILE, 'eval_data_file': MATRES_EVAL_DATA_FILE},
        'ptntime': {'model_name_or_path': PTNTIME_PRETRAINED_MODEL,
                    'output_dir': 'experiments/matres/ptntime',
                    'train_data_file': MATRES_TRAIN_DATA_FILE, 'eval_data_file': MATRES_EVAL_DATA_FILE}
    }
}