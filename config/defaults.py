from yacs.config import CfgNode as CN

_C = CN()

# Data
_C.DATA = CN()
_C.DATA.TRAIN_PATH = '~/data/imagenet2012/train/'
_C.DATA.TEST_PATH = '~/data/imagenet2012/val/'

# Experiment parameters
_C.RANDOM_SEED = 0
_C.NUM_RUNS = 5
_C.GPU_ID = 0
_C.N_WORKERS = 8

_C.MODEL = CN()
_C.MODEL.NAME = "ViTForImageClassification"
_C.MODEL.CONF = "google/vit-base-patch16-224"
_C.MODEL.WEIGHTS = None

# Experiment logging information
_C.EXPERIMENT_LOG = CN()
_C.EXPERIMENT_LOG.BASEPATH = 'results-reproduced/LiquidTransformers'
_C.EXPERIMENT_LOG.MODEL_NAME = str(_C.MODEL.NAME)
_C.EXPERIMENT_LOG.EXPERIMENT_NAME = "liquid_transformers"

# optimizer to use
_C.OPTIMIZER = CN()
_C.OPTIMIZER.EPOCHS = 20
_C.OPTIMIZER.BATCH_SIZE_TRAIN = 160
_C.OPTIMIZER.BATCH_SIZE_EVAL = 1600
_C.OPTIMIZER.LR = 1e-4

# Experiment logging information
_C.EXPERIMENT_LOG = CN()
_C.EXPERIMENT_LOG.BASEPATH = 'results-reproduced/LiquidTransformers'
_C.EXPERIMENT_LOG.MODEL_NAME = str(_C.MODEL.NAME)
_C.EXPERIMENT_LOG.EXPERIMENT_NAME = f"{_C.EXPERIMENT_LOG.MODEL_NAME}_lr_{_C.OPTIMIZER.LR}_epochs_{_C.OPTIMIZER.EPOCHS}_seed_{_C.RANDOM_SEED}"

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for foolingclip"""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
