# Thanks globus-compute !
try:
    from pydantic.v1 import *  # noqa: F401 F403
except ImportError:
    from pydantic import *  # type: ignore # noqa: F401 F403
