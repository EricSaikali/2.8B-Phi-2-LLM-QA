import os

assert os.path.isdir("model/checkpoints"), "You must have a directory named 'checkpoints' in the current directory."
assert os.path.isdir("model/models"), "You must have a directory named 'models' in the current directory."

files = [f for f in os.listdir('model/') if os.path.isfile('model/' + f)]
assert 'main_config.yaml' in files, "main_config.yaml not found in the current directory."
assert 'requirements.txt' in files, "requirements.txt not found in the current directory."
assert 'utils.py' in files, "utils.py not found in the current directory."

files = [f for f in os.listdir('model/models') if os.path.isfile('model/models/' + f)]
assert 'model_base.py' in files, "model_base.py not found in the current directory."
assert 'model_dpo.py' in files, "model_dpo.py not found in the current directory."


print('Model directory is correctly formatted.')