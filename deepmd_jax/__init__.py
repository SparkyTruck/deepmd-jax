import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('PJRT_GPU_ENABLE_ASYNC_DISPATCH', '1')
import jax
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# jax.config.update("jax_traceback_filtering", "off")
# jax.config.update("jax_log_compiles", True)
jax.config.update('jax_default_matmul_precision', 'float32')
np.set_printoptions(precision=4, suppress=True)

def _print(*args, **kwargs):
    if jax.process_index() == 0:
        print(*args, **kwargs)

_print('# DeepMD-jax: Starting on %d device(s):' % jax.device_count(), jax.devices())
if jax.process_count() > 1 and len({d.device_kind for d in jax.devices()}) > 1:
    raise ValueError('Multi-process simulation requires the same device kind on every process.')
from jax.sharding import Mesh
from jax.experimental import mesh_utils
mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), ('atom',))
jax.set_mesh(mesh)
