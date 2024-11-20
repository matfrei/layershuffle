import copy
import pickle as pkl
import os
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import argparse
from layershuffle.utils import get_cfg_defaults, Experiment
from train import train_model
from functools import partial
from pathlib import Path

# so how do we do this?
# easy, we parse the config the same way as we do for train, with the difference that we call a parameterized train, such the we can use hyperopt in this iteration to find all the paramters that we need.

# get optimization to start
# fix timestamp issue

def parse_args():
    parser = argparse.ArgumentParser(
        prog='find_hyperparams',
        description='Find good hyperparameters for a model and dataset',
        epilog='')
    parser.add_argument('--config-path', help='path to config file')
    args = parser.parse_args()
    return args

def train_run(params,config):
    config.OPTIMIZER.BATCH_SIZE_TRAIN = int(params['batch_size'])
    config.OPTIMIZER.LR = 10**params['lr']
    trainer = train_model(copy.deepcopy(config),save_model=False,use_bf=True)
    metrics = trainer.evaluate()
    loss = 1. - metrics['eval_accuracy']
    return {'loss': loss, 'status': STATUS_OK}
    

def find_hyperparams(config):
    experiment = Experiment(config.EXPERIMENT_LOG.BASEPATH, config.EXPERIMENT_LOG.MODEL_NAME, config.EXPERIMENT_LOG.EXPERIMENT_NAME,ts_dirname='hyperopt')
    outpath = os.path.join(experiment.modelpath,"params.pkl")
    Path(experiment.modelpath).mkdir(parents=True, exist_ok=True)
    space = {
        "lr":  hp.uniform('lr', -7, -1),
        "batch_size": hp.quniform("batch_size",8 ,320 , 8)}
    train_fn = partial(train_run,config=config)

    try:
        with open(outpath, 'rb') as f:
            (trials,best_accuracy) = pkl.load(f)
    except FileNotFoundError:
        trials = Trials()
        best_accuracy = 0.
    
    iter_evals = 1
    max_evals = iter_evals

    while True:
        best = fmin(fn=train_fn,
                space=space,
                algo=tpe.suggest, trials=trials,
                max_evals=max_evals)

        print("Best hyperparameters:", best)
        print("best accuracy", best_accuracy)
        with open(outpath, 'wb') as f:
            pkl.dump((trials,best_accuracy), f)
        max_evals += iter_evals

if __name__ == '__main__':
    config_path = parse_args().config_path
    config = get_cfg_defaults()
    config.merge_from_file(config_path)
    find_hyperparams(config)
