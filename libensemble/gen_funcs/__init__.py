import csv
import platform
from typing import Union, List, Optional


class RC:
    """Runtime configuration options."""

    _aposmm_optimizers: Optional[Union[str, List[str]]] = None  # optional string or list of strings
    _is_unix: bool = platform.system() in ["Linux", "Darwin"]

    @property
    def aposmm_optimizers(self):
        if self._is_unix:
            return self._aposmm_optimizers
        else:
            with open(".aposmm_opt.csv") as f:
                optreader = csv.reader(f)
                for opt in optreader:
                    return opt  # should only be one row

    @aposmm_optimizers.setter
    def aposmm_optimizers(self, values):
        self._aposmm_optimizers = values

        if not self._is_unix:
            with open(".aposmm_opt.csv", "w") as f:
                optwriter = csv.writer(f)
                if isinstance(values, list):
                    optwriter.writerow(values)
                else:
                    optwriter.writerow([values])


rc = RC()
