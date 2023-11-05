try:
    from .version import version as __version__
except ModuleNotFoundError:
    # if the user did `pip install -e .`, there will be no version.py file
    # therefore indicate that the version is "editable"
    __version__ = "editable"


# make some core functions available at the package level
import fastkde.fastKDE

pdf = fastkde.fastKDE.pdf
conditional = fastkde.fastKDE.conditional
pdf_at_points = fastkde.fastKDE.pdf_at_points

# import fastkde.plot if matplotlib is available
try:
    import fastkde.plot

    pair_plot = fastkde.plot.pair_plot
except ImportError:
    pass
