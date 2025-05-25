import sys
import torch
import torch.nn.functional as F


def plot_heatmap(df, vmin=None, vmax=None, filepath='../heatmap.pdf'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, annot=True, vmin=vmin, vmax=vmax, linewidths=0.5, ax=ax)
    plt.savefig(filepath, dpi=600)


def RMSELoss(yhat, y):
    import torch
    return torch.sqrt(torch.mean((yhat-y)**2))

# logger funcs


def log_args(args, log):
    for argname, argval in vars(args).items():
        log.info(f'{argname.replace("_"," ").capitalize()}: {argval}')


def add_args(args):
    import torch
    # check model type and add feature_size
    from available_models import all_models
    args.feature_size = all_models[args.pretrained_model]['feature_size']
    
    # dataparallel
    args.dataparallel = False
    if torch.cuda.device_count() > 1:
        args.dataparallel = True
    return args


class LoggerWritter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

# modify state dict from data parallel


def modify_dict_from_dataparallel(state_dict, args):
    if args.dataparallel:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

    
    


#FROM HERE, Evidential deep learning

# --------- EDL LOSS FUNCTION ---------
def edl_mse_loss(pred_alpha, target, epoch=1, num_classes=2, coeff=1.0):
    S = torch.sum(pred_alpha, dim=1, keepdim=True)
    one_hot = F.one_hot(target, num_classes).float()
    loss = torch.sum((one_hot - pred_alpha / S) ** 2, dim=1, keepdim=True)
    reg = coeff * torch.sum((pred_alpha - 1) ** 2, dim=1, keepdim=True)
    return torch.mean(loss + reg)


# --------- CALIBRATION METRIC ---------
def compute_ece(probabilities, labels, n_bins=10):
    confidences, predictions = probabilities.max(1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probabilities.device)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            accuracy = accuracies[mask].float().mean()
            confidence = confidences[mask].mean()
            ece += (confidence - accuracy).abs() * mask.float().mean()

    return ece.item()
