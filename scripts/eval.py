import os, argparse
import torch
import pandas as pd

from layershuffle.utils import Experiment, parse_preprocessor, parse_model, parse_dataset, get_cfg_defaults, load_model_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def parse_args():
    parser = argparse.ArgumentParser(
        prog='eval',
        description='Eval a model',
        epilog='')
    parser.add_argument('--config-path', help='path to config file')
    args = parser.parse_args()
    return args

def eval_model():
    
    config_path = parse_args().config_path
    config = get_cfg_defaults()
    config.merge_from_file(config_path)
    config.freeze()

    device = torch.device(f'cuda:{config.GPU_ID}') if torch.cuda.is_available() else torch.device("cpu")
    processor = parse_preprocessor(config)
    # TODO: should be the test case in that case, right?
    train_set,val_set = parse_dataset(config,processor)
    loader = torch.utils.data.DataLoader(val_set, batch_size=config.OPTIMIZER.BATCH_SIZE_EVAL, num_workers=config.N_WORKERS)
    scores = []

    experiment = Experiment(config.EXPERIMENT_LOG.BASEPATH, config.EXPERIMENT_LOG.MODEL_NAME, config.EXPERIMENT_LOG.EXPERIMENT_NAME,ts_dirname=config.EXPERIMENT_LOG.TIMESTAMP)
    for seed in range(config.RANDOM_SEED,config.RANDOM_SEED+config.NUM_RUNS):
        # TODO: how to load the correct model weights?
        # A possible way is to just take the corresponding directory as read out from the config, and evalutate all runs in the directory... maybe the timestamp should be fixed then
        # let's just make a timestamp property in the config that is ignored for train but used for eval. (that way we have also documented in the github which model we ran on the harddrive)
        model = parse_model(config)
        model_path = os.path.join(experiment.modelpath,f"run_{seed}","model.safetensors")
        load_model_weights(model,model_path)
        model.to(device)
        
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(loader):
                images = images.to(device)
                target = target.to(device)
                outputs = model(images)
                logits = outputs.logits
                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                #print(acc1)
                top1 += acc1
                top5 += acc5
                n += images.size(0)
            print(n)
            print(top1)
            top1 = (top1 / n) * 100
            top5 = (top5 / n) * 100

        print(f"Top-1 accuracy: {top1:.2f}")
        print(f"Top-5 accuracy: {top5:.2f}")
        scores.append((top1,top5))

    json = pd.DataFrame(scores, columns=['top1_acc', 'top5_acc']).to_json()
    if config.MODEL.SHUFFLE:
        postfix = 'arbitrary'
    else:
        postfix = 'sequential'
        
    output_dir=os.path.join(experiment.modelpath,f"{config.MODEL.NAME}_scores_{postfix}.json")
    with open(output_dir,"w") as file:
        file.write(json)

        
if __name__ == '__main__':
    eval_model()
