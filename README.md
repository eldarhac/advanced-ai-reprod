Running prerequisites:

- [x] Python 3.8
- [x] GCC 11.4.0
- [x] GPU with CUDA 10 or higher
- [x] 32 GB VRAM 
- [x] 64 GB RAM
- [x] 100 GB of free disk space

See reproduction example in `reproduction_notebook.ipynb`


# Files
- `config.py` - configuration of all the datasets and model weights paths
- `training_args.py` - arguments for training the models
- `evaluation.py` - evaluation logic for the model predictions
- `models.py` - the different models (PtnTime/SymTime etc.) configurations
- `train_model.py` - code for training and evaluating model
- `main.py` - a main file for running all experiments
- `reproduction_notebook.ipynb` - a reproduction example in a jupyter notebook.
- `/data` - a directory with all the datasets used in the reproduction
