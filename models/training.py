
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score

from ignite.engine import Events
from ignite.metrics import MeanSquaredError

import logging
import numpy as np

from engines import create_supervised_trainer, create_supervised_evaluator
from handlers import EarlyStopping
from utils import create_dir


logger = logging.getLogger(__name__)


def get_optimiser(opt_name, model_parameters, opt_parameters):

    lr = opt_parameters.get('lr', 0.01)
    wd = opt_parameters.get("weight_decay", 0.0)

    if opt_name.lower() == 'sgd':
        optimiser = optim.SGD(model_parameters, lr=lr, weight_decay=wd,
                              momentum=opt_parameters.get('momentum', 0),
                              dampening=opt_parameters.get('dampening', 0),
                              nesterov=opt_parameters.get('nesterov', False))

    elif opt_name.lower() == 'adam':
        optimiser = optim.Adam(model_parameters, lr=lr, weight_decay=wd,
                               betas=opt_parameters.get('betas', (0.9, 0.999)))

    elif opt_name.lower() == 'rmsprop':
        optimiser = optim.RMSprop(model_parameters, lr=lr, weight_decay=wd,
                                  momentum=opt_parameters.get('momentum', 0),
                                  alpha=opt_parameters.get('alpha', 0.99))
    else:
        raise NotImplementedError(opt_name + ' optmiser not supported.')

    return optimiser


def train_network(
    net, train_loader, valid_loader, hparameters, device, dtype,
    loggers=[False, False, False], log_dir=None):
    """
    Network trainer using the ignite framework.

    Args:
        net (`torch.nn.Module`): the model to be trained.
        train_loader (`torch.data.DataLoader`): data loader for the training set.
        valid_loader (`torch.data.DataLoader`): data loader for the validation set.
        hparameters (dict): hyper-parameters for the training process.
        device (`torch.device`): device on which training and evaluation are performed.
        dtype (`torch.dtype`): data type for the tensors under processing.
        loggers (list of len 3): verbosity of the trainer.
        
    TODO:
        - At the moment, tensorboard logs are commented.
        - There is a discrepancy with the training loss and the MSE metric, as the implementation
          in the pytorch core and in ignite are not consistent.
    """

    # Define loss and optimizer
    criterion = nn.MSELoss(reduction=hparameters.get('mse_reduction', 'mean'))
    metrics = {'mse': MeanSquaredError()}

    optimizer = get_optimiser(hparameters.get('optimiser', 'adam'), net.parameters(), hparameters)

    # define training and evaluation engines
    trainer = create_supervised_trainer(net, optimizer, criterion, device, dtype)
    train_evaluator = create_supervised_evaluator(net, metrics, device, dtype)
    valid_evaluator = create_supervised_evaluator(net, metrics, device, dtype)

    # adding early stopping criterion
    def score_function(engine):
        val_loss = engine.state.metrics['mse']
        return -val_loss
    es_handler = EarlyStopping(
        patience=hparameters.get('patience', 20),
        score_function=score_function, trainer=trainer, model=net)
    # the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    valid_evaluator.add_event_handler(Events.COMPLETED, es_handler, net)

    # keep track of the training and validation loss
    tr_history = {'training_loss': [], 'validation_loss': []}
    # create_dir(log_dir)  # creating the log dir if not None
    # writer = create_summary_writer(net, train_loader, log_dir)

    if(loggers[0]):
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            print("Epoch {} | Batch {} | Loss: {:.2f}".format(
                trainer.state.epoch, (trainer.state.iteration - 1) % len(train_loader), trainer.state.output))
            # writer.add_scalar("training/loss", trainer.state.output, trainer.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        tr_history['training_loss'].append(metrics['mse'])
        if(loggers[1]):
            # writer.add_scalars('MSE', {"training": metrics['mse']}, trainer.state.epoch)
            print("Epoch: {} - Training loss: {:.2f} | MSE: {:.2f}"
                  .format(trainer.state.epoch, trainer.state.output, metrics['mse']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        valid_evaluator.run(valid_loader)
        metrics = valid_evaluator.state.metrics
        tr_history['validation_loss'].append(metrics['mse'])
        if(loggers[2]):
            # writer.add_scalars('MSE', {"validation": metrics['mse']}, trainer.state.epoch)
            print("Epoch: {} - Validation MSE: {:.2f}"
                  .format(trainer.state.epoch, metrics['mse']))

    trainer.run(train_loader, max_epochs=hparameters.get("num_epochs", 10000))

    # writer.close()
    return es_handler.get_best_model_after_stop(), tr_history


class RegressionModelEvaluator():

    def __init__(self, model, device, dtype):
        """
        Class-constructor for the evaluator.

        Args:
            model (`torch.Module`): the torch model to evaluate.
            device (`torch.device`): device on which the eval is performed.
            dtype (`torch.dtype`): data type for the tensors under processing.

        """

        self.model = model
        self.dtype = dtype
        self.device = device


    def evaluate_net_raw(self, test_loader, all_sq_errors=False):
        """
        Runner function: returns the actual evaluation of the model.

        Args:
            test_loader (`data.DataLoader`): data loader for the test set.
            all_sq_errors (boolean): false if mean and sqrt of the squared errors
                for each observation has to be computed, otherwise a numpy matrix
                containing the squared error for each test entry is returned.
        """

        self.model.eval()
        criterion_per = nn.MSELoss(reduction='none')
        targets, outputs = [], []

        with torch.no_grad():
            for xdata, target in test_loader:
                # getting model's predictions on the test data
                xdata = xdata.to(device=self.device, dtype=self.dtype)
                targets.append(target.to(device=self.device, dtype=self.dtype))
                outputs.append(self.model(xdata))
    
        targets = torch.cat(targets, 0)
        outputs = torch.cat(outputs, 0)
        
        # computing the squared error for each output feature (no agg)    
        squared_error = criterion_per(outputs, targets)
        if not all_sq_errors:
            squared_error = torch.sqrt(torch.mean(torch.as_tensor(squared_error), 0))
        r2score = r2_score(
            targets.cpu().numpy(), outputs.cpu().numpy(), multioutput='raw_values')
        
        return squared_error.cpu().numpy(), r2score, targets.cpu(), outputs.cpu()


def allsets_regressor_evaluation(
    model, tr_loader, va_loader, te_loader, device=None, dtype=torch.double, prints=False):

    eval_setup = {'model': model, 'device': device, 'dtype': dtype}
    rme = RegressionModelEvaluator(**eval_setup)

    tr_rmse, tr_r2score  = rme.evaluate_net_raw(tr_loader)
    va_rmse, va_r2score  = rme.evaluate_net_raw(va_loader)
    te_rmse, te_r2score  = rme.evaluate_net_raw(te_loader)

    return (tr_rmse, tr_r2score), (va_rmse, va_r2score), (te_rmse, te_r2score)    
    

def create_summary_writer(model, data_loader, log_dir):
    """
    Summary writer factory for tensorboard viz.
    """

    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)

    try:
        writer.add_graph(model, x.to(next(model.parameters()).device))
    except Exception as e:
        logger.warning("Failed to save model graph: {}".format(e))

    return writer



