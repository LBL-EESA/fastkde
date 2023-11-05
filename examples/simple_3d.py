from fastkde import fastKDE
from scipy import stats
import pylab as PP
import matplotlib as mpl
from numpy import *

# set plot default fonts (fonts that are generally nice figures
font = {"family": "serif", "size": "15", "weight": "bold"}
mpl.rc("font", **font)
mpl.rc(
    "axes", labelweight="bold"
)  # needed for bold axis labels in more recent version of matplotlib


N = int(1e3)  # number of points

# generate 3 independent samples from 3 different distributions
x_1 = stats.norm.rvs(size=N)
x_2 = stats.gamma.rvs(2, size=N)
x_3 = stats.betaprime.rvs(5, 6, size=N)

# calculate the 3D PDF
pdf, values = fastKDE.pdf(
    x_1, x_2, x_3, numPoints=[65, 65, 65]
)  # simply add more variables to the argument list for higher dimensions
# note though that memory quickly becomes an issue
# the numPoints argument results in a coarser PDF--but one that is calculated
# faster (and with less memory)

# calculate the index of the mode of the distribution
# (we'll plot 2D slices through the mode)
i_mode_ravel = argmax(pdf.ravel())
nmode = unravel_index(i_mode_ravel, shape(pdf))

# set the levels
clevels = linspace(0, pdf[nmode], 64)

# create the plot
fig, axs = PP.subplots(1, 3, figsize=(15, 5))

# plot slices across the mode of the distribution
ax = axs[0]
ax.contourf(values[0], values[1], pdf[nmode[0], :, :], levels=clevels)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")

ax = axs[1]
ax.contourf(values[0], values[2], pdf[:, nmode[1], :], levels=clevels)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_3$")

ax = axs[2]
ax.contourf(values[1], values[2], pdf[:, :, nmode[2]], levels=clevels)
ax.set_xlabel("$x_2$")
ax.set_ylabel("$x_3$")

PP.tight_layout()  # fix subpanel spacing
PP.show()
