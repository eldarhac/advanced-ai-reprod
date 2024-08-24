from config import config_dict
from evaluation import evaluate_tracie_style, evaluate_symbolic
from train_model import run_and_eval


run_and_eval(**config_dict['iid']['base_model'])
print('evaluating iid base model:')
evaluate_tracie_style(config_dict['iid']['base_model']['eval_data_file'], config_dict['iid']['base_model']['output_dir'])
print('='*50)

run_and_eval(**config_dict['iid']['ptntime'])
print('evaluating iid ptntime:')
evaluate_tracie_style(config_dict['iid']['ptntime']['eval_data_file'], config_dict['iid']['ptntime']['output_dir'])
print('='*50)

run_and_eval(**config_dict['iid']['symtime'], per_gpu_train_batch_size=4,
                 per_device_train_batch_size=4, gradient_accumulation_steps=2,
                 per_device_eval_batch_size=4, per_gpu_eval_batch_size=4)
print('evaluating iid symtime:')
evaluate_symbolic(config_dict['iid']['symtime']['eval_data_file'], config_dict['iid']['symtime']['output_dir'])
print('='*50)

run_and_eval(**config_dict['uniform']['base_model'])
print('evaluating uniform base model:')
evaluate_tracie_style(config_dict['uniform']['base_model']['eval_data_file'], config_dict['uniform']['base_model']['output_dir'])
print('='*50)

run_and_eval(**config_dict['uniform']['ptntime'])
print('evaluating uniform ptntime:')
evaluate_tracie_style(config_dict['uniform']['ptntime']['eval_data_file'], config_dict['uniform']['ptntime']['output_dir'])
print('='*50)

run_and_eval(**config_dict['uniform']['symtime'], per_gpu_train_batch_size=4,
                    per_device_train_batch_size=4, gradient_accumulation_steps=2,
                    per_device_eval_batch_size=4, per_gpu_eval_batch_size=4)
print('evaluating uniform symtime:')
evaluate_symbolic(config_dict['uniform']['symtime']['eval_data_file'], config_dict['uniform']['symtime']['output_dir'])
print('='*50)

run_and_eval(**config_dict['matres']['base_model'])
print('evaluating matres base model:')
evaluate_tracie_style(config_dict['matres']['base_model']['eval_data_file'], config_dict['matres']['base_model']['output_dir'])
print('='*50)

run_and_eval(**config_dict['matres']['ptntime'])
print('evaluating matres ptntime:')
evaluate_tracie_style(config_dict['matres']['ptntime']['eval_data_file'], config_dict['matres']['ptntime']['output_dir'])
print('='*50)