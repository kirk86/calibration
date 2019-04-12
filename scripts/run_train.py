import subprocess
from utils.parameter import AppConfig, ModelParams

config = AppConfig('./settings/config.yaml', 'appconfig')
params = ModelParams('./settings/params.yaml', 'params')

for dataset in params.datasets:
    for net in params.networks:
        subprocess.call([
            'python', 'app.py', '-n', dataset + '.' + net, '-m', 'MY_DB',
            'with', 'default-params.params.num_epochs=5',
            'default-params.params.datasets={}'.format(dataset),
            'default-params.params.networks={}'.format(net)
        ])
