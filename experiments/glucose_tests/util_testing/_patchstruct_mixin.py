import torch
from util_testing.patchwise_structural import PSLoss_GDW
from neuralforecast.auto import AutoNHITS, AutoNBEATS

class GDWLossMixin:
    """
    A mixin class for NeuralForecast models (inheriting from BaseModel)
    to enable Gradient-based Dynamic Weighting with PSLoss_GDW.

    This mixin overrides the `backward` step. It checks if the
    model's loss function is `PSLoss_GDW`. If so, it intercepts
    the "dummy" loss from the training_step, computes the
    dynamically weighted loss, and performs the backward pass.

    This class must be inherited *before* the main model class, e.g.:
    `class NHITS_GDW(GDWLossMixin, NHITS): ...`
    """

    @property
    def _gdw_output_layer(self) -> torch.nn.Module:
        """
        Dynamically finds the output layer for GDW gradient calculation.
        This is a best-effort attempt based on common names in NeuralForecast.
        
        If your custom model uses a different name, override this
        property in your final class.
        
        Example:
        `class MyModel_GDW(GDWLossMixin, MyModel):`
        `    @property`
        `    def _gdw_output_layer(self):`
        `        return self.my_final_layer`
        """
        # Common output layer names in NeuralForecast models
        common_names = [
            'output_projection', # NHITS
            'head',              # PatchTST
            'linear',            # DLinear, NLinear
            'c_out',             # TimesNet
            'flatten_and_linear' # TSMixer
        ]
        
        for name in common_names:
            if hasattr(self, name):
                layer = getattr(self, name)
                if isinstance(layer, torch.nn.Module):
                    return layer
        
        raise AttributeError(
            f"GDWLossMixin could not automatically find the output layer. "
            f"Tried {common_names}. Please override the `_gdw_output_layer` "
            f"property in your model class to return the correct output layer."
        )

    def backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        """
        Overrides the default LightningModule `backward` step.
        """
        # Check if we are using the GDW loss.
        # self.loss is the loss instance (e.g., MSE(), PSLoss_GDW())
        if isinstance(self.loss, PSLoss_GDW):
            print('proper use')
            # `loss` passed in is the "dummy" L_mse. We ignore it.
            # We now compute the *real* total loss using the
            # components stored in self.loss.
            try:
                output_layer = self._gdw_output_layer
            except AttributeError as e:
                raise AttributeError(
                    f"GDWLossMixin requires a valid output layer. {e}"
                ) from e

            # This computes GDW weights and returns the final weighted tensor
            total_loss = self.loss.compute_gdw_total_loss(output_layer)
            
            # Perform the backward pass on the *actual* total loss
            super().backward(total_loss, *args, **kwargs)
            
            # Log GDW components
            # We use `on_step=True, on_epoch=False` for granular (but noisy) logging
            # Set to `on_epoch=True` for cleaner end-of-epoch averages
            self.log('train_L_ps', self.loss.L_ps_weighted_detached.detach(), on_step=True, on_epoch=False, prog_bar=False)
            self.log('train_alpha', self.loss.alpha_detached, on_step=True, on_epoch=False, prog_bar=False)
            self.log('train_beta', self.loss.beta_detached, on_step=True, on_epoch=False, prog_bar=False)
            self.log('train_gamma', self.loss.gamma_detached, on_step=True, on_epoch=False, prog_bar=False)
            # The 'train_loss' (total loss) will be logged by the BaseModel's training_step
            # from the 'dummy' L_mse, which is fine, but we can log the real one too.
            self.log('train_loss_total', total_loss.detach(), on_step=True, on_epoch=False, prog_bar=True)


        else:
            # Not our special loss, proceed as normal
            # This calls the original LightningModule.backward()
            super().backward(loss, *args, **kwargs)
class AutoNHITS_GDW(GDWLossMixin, AutoNHITS):
    """ AutoNHITS with PSLoss_GDW support via GDWLossMixin. """
    def __init__(self, **kwargs):
        # Ensure the loss is PSLoss_GDW if provided, otherwise default might break mixin
        if 'loss' in kwargs and not isinstance(kwargs['loss'], PSLoss_GDW):
             print(f"Warning: Initializing AutoNHITS_GDW with loss type {type(kwargs['loss'])}. "
                   f"GDW backward step will only work if loss is PSLoss_GDW.")
        elif 'loss' not in kwargs:
             raise ValueError("AutoNHITS_GDW requires a PSLoss_GDW instance during initialization.")
        super().__init__(**kwargs)


class AutoNBEATS_GDW(GDWLossMixin, AutoNBEATS):
    """ AutoNBEATS with PSLoss_GDW support via GDWLossMixin. """
    def __init__(self, **kwargs):
        if 'loss' in kwargs and not isinstance(kwargs['loss'], PSLoss_GDW):
             print(f"Warning: Initializing AutoNBEATS_GDW with loss type {type(kwargs['loss'])}. "
                   f"GDW backward step will only work if loss is PSLoss_GDW.")
        elif 'loss' not in kwargs:
             raise ValueError("AutoNBEATS_GDW requires a PSLoss_GDW instance during initialization.")
        super().__init__(**kwargs)