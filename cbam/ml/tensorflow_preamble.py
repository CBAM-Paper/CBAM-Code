
class TensorflowConfig:
    called = False

    @classmethod
    def configure_tensorflow(cls, gpu_num: int = None):
        import os
        from cbam.utils.config import Config
        if not cls.called:
            os.environ['TF_XLA_FLAGS'] = "--tf_xla_enable_xla_devices"
            if gpu_num is None:
                gpu_num = Config.get_gpu_num()
            # Only use one GPU
            if int(gpu_num) >= 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
                print(f"Using GPU {gpu_num}!")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
        cls.called = True
