import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ipu"))
sys.path.insert(0, str(ROOT / "ipu" / "kernels"))
import emu  # noqa
from mlkem_consts import Q  # noqa

def vec(name, dtype=np.int16, shape=None):
    a = np.fromfile(ROOT / "vectors" / f"{name}.bin", dtype=dtype)
    return a.reshape(shape) if shape else a

def modq(x):
    return np.mod(np.rint(np.asarray(x, np.float64)).astype(np.int64), Q)
