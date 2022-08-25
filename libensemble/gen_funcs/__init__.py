import os
import csv
import time
import logging
import platform
from typing import Union, List, Optional

logger = logging.getLogger(__name__)


class RC:
    """Runtime configuration options."""

    _aposmm_optimizers: Optional[
        Union[str, List[str]]
    ] = None  # optional string or list of strings
    _is_unix: bool = platform.system() in ["Linux", "Darwin"]
    _csv_path = __file__.rsplit("/", 1)[0] + "/.aposmm_opt.csv"

    @property
    def aposmm_optimizers(self):
        timeout = 5
        start = time.time()
        if self._is_unix:
            return self._aposmm_optimizers
        else:
            while not os.path.isfile(self._csv_path):
                time.sleep(0.2)
                if time.time() - start > timeout:
                    logger.warning(
                        "Unable to determine set optimization methods by timeout. Using nlopt as default."
                    )
                    return "nlopt"

            with open(self._csv_path) as f:
                optreader = csv.reader(f)
                for opt in optreader:
                    return opt  # should only be one row

    @aposmm_optimizers.setter
    def aposmm_optimizers(self, values):
        self._aposmm_optimizers = values

        if not self._is_unix and not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w") as f:
                optwriter = csv.writer(f)
                if isinstance(values, list):
                    optwriter.writerow(values)
                else:
                    optwriter.writerow([values])


rc = RC()
