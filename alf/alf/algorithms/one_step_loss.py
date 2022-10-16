

import alf
from alf.utils import losses
from alf.algorithms.td_loss import TDLoss


@alf.configurable
class OneStepTDLoss(TDLoss):
    def __init__(self,
                 gamma=0.98,
                 td_lambda=0.0,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 debug_summaries=False,
                 name="TDLoss"):
        """
        Args:
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            debug_summaries (bool): True if debug summaries should be created
            name (str): The name of this loss.
        """
        super().__init__(
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            debug_summaries=debug_summaries,
            td_lambda=td_lambda,
            name=name)
