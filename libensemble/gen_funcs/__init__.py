import platform
from dataclasses import dataclass
from typing import Union, List, Optional


@dataclass
class rc:
    """Runtime configuration options."""

    aposmm_optimizers: Optional[
        Union[str, List[str]]
    ] = None  # optional string or list of strings
    is_unix: bool = platform.system() in ["Linux", "Darwin"]
