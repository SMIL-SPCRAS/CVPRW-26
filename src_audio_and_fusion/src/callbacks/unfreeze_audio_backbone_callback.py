from chimera_ml.core.registry import CALLBACKS
from chimera_ml.callbacks.base import BaseCallback


class UnfreezeAudioBackboneCallback(BaseCallback):
    """
    At a given epoch, unfreezes last N layers of model backbone (if supported).
    Works with your audeering_wav2vec2_model_v2 / hf_wav2vec2_va_model if it exposes:
      - model.unfreeze_last_n_layers(n)
      - or model.unfreeze_backbone()
    """

    def __init__(self, unfreeze_epoch: int = 3, unfreeze_last_n_layers: int = 4):
        super().__init__()
        self.unfreeze_epoch = int(unfreeze_epoch)
        self.unfreeze_last_n_layers = int(unfreeze_last_n_layers)
        self._done = False

    def on_train_epoch_start(self, trainer, pl_module, epoch: int, **kwargs):
        if self._done:
            return
        if epoch != self.unfreeze_epoch:
            return

        model = getattr(trainer, "model", None) or getattr(trainer, "module", None) or pl_module
        if hasattr(model, "unfreeze_last_n_layers"):
            model.unfreeze_last_n_layers(self.unfreeze_last_n_layers)
        elif hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()
        else:
            raise RuntimeError("Model does not support unfreezing (no unfreeze_last_n_layers/unfreeze_backbone).")

        self._done = True
        trainer.logger.info(
            f"[UnfreezeAudioBackboneCallback] Unfroze last {self.unfreeze_last_n_layers} layers at epoch={epoch}"
        )


@CALLBACKS.register("unfreeze_audio_backbone_callback")
def unfreeze_audio_backbone_callback(**params):
    return UnfreezeAudioBackboneCallback(**params)
