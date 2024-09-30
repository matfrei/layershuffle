# LayerShuffle: Enhancing Robustness in Vision Transformers by Randomizing Layer Execution Order

Installation
-------
```
pip install -e .
```

Training and evaluating models
-------

To train models use
```
python scripts/train.py --config-path config/<config-name>.yaml
```

To evaluate trained models note the timestamp of the model written to \<outputdir\>/\<model_name\>/\<experimentname\>/\<timestamp\>
and add it in the corresponding configuration file under EXPERIMENTLOG -> TIMESTAMP.
E.g., if your timestamp is 17-09-2024_03-39-44, the corresponding section of your config file should look as follows

```
EXPERIMENT_LOG:
  BASEPATH: "./results/"
  MODEL_NAME: "DeiTForImageClassification"
  EXPERIMENT_NAME: "CIFAR100_lr_1e-4_epochs_100"
  TIMESTAMP: "17-09-2024_03-39-44"
```
Then run
```
python scripts/eval.py --config-path config/<config-name>.yaml
```
