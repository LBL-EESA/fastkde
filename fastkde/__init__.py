try:
    from .version import version as __version__
except ModuleNotFoundError:
    # if the user did `pip install -e .`, there will be no version.py file
    # therefore indicate that the version is "editable"
    __version__ = "editable"
