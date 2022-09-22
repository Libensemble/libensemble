import os
import csv
import logging

logger = logging.getLogger(__name__)


class RC:
    """Runtime configuration options."""

    def __init__(self):

        self._opt_modules = None  # optional string or list of strings
        self._csv_path: str = __file__.replace("__init__.py", ".opt_modules.csv")
        self._csv_exists: bool = os.path.isfile(self._csv_path)

        if self._csv_exists:
            with open(self._csv_path) as f:
                optreader = csv.reader(f)
                self._opt_modules = [opt for opt in optreader][0]  # should only be one row

    @property  # getter
    def aposmm_optimizers(self):
        if self._opt_modules or not self._csv_exists:
            return self._opt_modules
        else:
            with open(self._csv_path) as f:
                optreader = csv.reader(f)
                return [opt for opt in optreader][0]

    @aposmm_optimizers.setter
    def aposmm_optimizers(self, values):

        current_opt = self.aposmm_optimizers
        if not isinstance(values, list):
            values = [values]

        if self._csv_exists and values != current_opt:  # avoid rewriting constantly
            with open(self._csv_path, "w") as f:
                optwriter = csv.writer(f)
                optwriter.writerow(values)

        self._opt_modules = values


rc = RC()
