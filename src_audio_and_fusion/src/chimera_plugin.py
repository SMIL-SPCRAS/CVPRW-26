def register() -> None:
    import audio.data.abaw_va_datamodule  
    import audio.data.abaw_va_test_datamodule  
    import audio.models.wavlm_s2s_model
    import losses.ccc_mse_loss  
    import metrics.va_ccc_metric  
    
    import callbacks.framewise_eval_callback
    import callbacks.windowwise_callback
    import callbacks.unfreeze_audio_backbone_callback
    import callbacks.ema_callback

    import optimizers.adamw_two_group_optimizer 
    import schedulers.named_reduceonplateau_scheduler


    import fusion.data.abaw_mm_datamodule
    import fusion.data.abaw_mm_test_datamodule
    import fusion.models.multimodal_fusion_model
    import losses.ccc_mse_mm_loss
    import callbacks.multimodal_framewise_callback
    import callbacks.multimodal_framewise_v2_callback