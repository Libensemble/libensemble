import logging

logger = logging.getLogger(__package__ )
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('ensemble.log')
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

#logger.debug("Testing top level logger")

def set_level(level):
    numeric_level = getattr(logging, level.upper(), 10)
    logger.setLevel(numeric_level)
