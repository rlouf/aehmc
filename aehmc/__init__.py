from typing import Callable, Dict, NewType, Optional, Tuple

from aesara.tensor.var import TensorVariable

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


KernelType = Callable[
    [
        TensorVariable,
        TensorVariable,
        TensorVariable,
        TensorVariable,
        TensorVariable,
        ...,
    ],
    Tuple[
        Tuple[TensorVariable, ...],
        Dict,
    ],
]
