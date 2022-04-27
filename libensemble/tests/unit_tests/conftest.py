# https://stackoverflow.com/questions/47559524/pytest-how-to-skip-tests-unless-you-declare-an-option-flag/61193490#61193490

import pytest


def pytest_addoption(parser):
    parser.addoption("--runextra", action="store_true", default=False, help="run extra tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "extra: mark test as extra to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runextra"):
        # --runextra given in cli: do not skip extra tests
        return
    skip_extra = pytest.mark.skip(reason="need --runextra option to run")
    for item in items:
        if "extra" in item.keywords:
            item.add_marker(skip_extra)
