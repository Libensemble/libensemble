import os
import csv
import time
import logging
import platform
from typing import Union, List, Optional

logger = logging.getLogger(__name__)


class RC:
    """Runtime configuration options."""

    def __init__(self):

        self._opt_modules: Optional[Union[str, List[str]]] = None  # optional string or list of strings
        self._is_windows: bool = platform.system() not in ["Linux", "Darwin"]
        self._csv_path = __file__.replace("__init__.py", ".opt_modules.csv")
        self._csv_exists: bool = os.path.isfile(self._csv_path)

        if self._is_windows and self._csv_exists:
            with open(self._csv_path) as f:
                optreader = csv.reader(f)
                self._opt_modules = [opt for opt in optreader][0]  # should only be one row

    @property
    def aposmm_optimizers(self):
        timeout = 5
        start = time.time()
        if not self._is_windows or not self._csv_exists:
            return self._opt_modules
        else:  # is windows and csv exists
            while not os.path.isfile(self._csv_path):  # self._csv_exists not updated after init
                time.sleep(0.1)
                if time.time() - start > timeout:
                    logger.warning("Unable to determine set optimization methods by timeout. Using nlopt as default.")
                    return "nlopt"

            while not os.stat(self._csv_path).st_size:
                time.sleep(0.001)  # avoiding race where file may exist but values not written into it yet

            with open(self._csv_path) as f:
                optreader = csv.reader(f)
                return [opt for opt in optreader][0]

    @aposmm_optimizers.setter
    def aposmm_optimizers(self, values):

        current_opt = self.aposmm_optimizers
        if not isinstance(values, list):
            values = [values]

        if self._is_windows and self._csv_exists and values != current_opt:  # avoid rewriting constantly
            with open(self._csv_path, "w") as f:
                optwriter = csv.writer(f)
                optwriter.writerow(values)

        self._opt_modules = values


rc = RC()
