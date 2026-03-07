import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import jax
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# jax.config.update("jax_traceback_filtering", "off")
# jax.config.update("jax_log_compiles", True)
jax.config.update('jax_default_matmul_precision', 'float32')
np.set_printoptions(precision=4, suppress=True)
print('# DeepMD-jax: Starting on %d device(s):' % jax.device_count(), jax.devices())
from jax.sharding import Mesh
from jax.experimental import mesh_utils
mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), ('atom',))
jax.set_mesh(mesh)