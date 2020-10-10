
import copy
import logging

from ignite.engine import Engine


class EarlyStopping(object):
    """
    EarlyStopping handler can be used to stop the training if no improvement
    after a given number of events. This is custom version of the original
    handler provided in the ignite framework, where the state dictionary
    of the model under training is saved at each loss improvement.

    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training.
        score_function (callable):
            It should be a function taking a single argument,
            an :class:`~ignite.engine.Engine` object, and return a score `float`.
            An improvement is considered if the score is higher.
        trainer (Engine):
            trainer engine to stop the run if no improvement.
    
    TODO:
        - Set a delta (minimum increment) before reinitialising the counter.
    """

    def __init__(self, patience, score_function, trainer, model):

        if not callable(score_function):
            raise TypeError("Argument score_function should be a function.")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if not isinstance(trainer, Engine):
            raise TypeError("Argument trainer should be an instance of Engine.")

        self.score_function = score_function
        self.patience = patience
        self.trainer = trainer
        self.counter = 0

        self.best_score = None
        self.model = copy.deepcopy(model)
        self.best_mdict = model.state_dict()

        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())

    def __call__(self, engine, model):
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            self._logger.debug("EarlyStopping: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self._logger.info("EarlyStopping: Stop training")
                self.trainer.terminate()
        else:
            self.best_mdict = copy.deepcopy(model.state_dict())
            self.best_score = score
            self.counter = 0

    def get_best_model_after_stop(self):
        """
        Returns the best loss during training, -patience epochs before the stop.
        """
        if self.counter < self.patience:
            self._logger.warn("EarlyStopping: training is not stopped yet.")
        self.model.load_state_dict(self.best_mdict)

        return self.model, -self.best_score
