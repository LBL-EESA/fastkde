import numpy as np
import pylab as PP
import fastkde.fastKDE as fastKDE

def cumulative_integral(pdf, axes, integration_axes=None, reverse_axes=None):

    """
    Calculates the cumulative integral of the pdf, given the axis values
    
       input:
       ------
           pdf              : a PDF (presumably from the .pdf member of a fastKDE object)
           axes             : a list of PDF axes (presumably from the .axes member of a fastKDE object)
           integration_axes : the axes along which to integrate (default is all).  These have the same
                              ordering as axes in the axes input variable.
           reverse_axes     : axes along which to reverse the direction of the cumulative calculation
           
       output:
       -------
           returns an array with the same shape as PDF, but with the integral calculated cumulatively
    """

    #Set the default value for integration_axes
    if integration_axes is None:
        integration_axes = range(len(np.shape(pdf)))
    
    #Set the rank of the pdf
    rank = len(np.shape(pdf))
    
    #Set the default value for reverse_axes
    if reverse_axes is None:
        reverse_axes = []
    
    #Make sure that axes is an iterable of iterables
    try:
        axes[0][0]
    except:
        axes = [axes]
    
    #Make sure that reverse_axes is an iterable
    try:
        len(reverse_axes)
    except:
        reverse_axes = [reverse_axes]
    
    #Make sure that integration_axes is an iterable
    try:
        len(integration_axes)
    except:
        integration_axes = [integration_axes]
    
    #Copy the pdf array
    cpdf = np.array(pdf)
    
    #Loop over the axes for which we are calculating the cumulative
    for i in integration_axes:
        #Calculate the grid spacing; assume it is uniform
        dx = np.diff(axes[i])
        dx = np.concatenate([dx,[dx[-1]]])


        #Calculate the index of pdf corresponding to axis i
        iprime = rank - i - 1

        # broadcast dx to the ~same shape as cpdf
        broadcast_tuple = len(cpdf.shape)*[np.newaxis]
        broadcast_tuple[iprime] = slice(None,None,None)
        broadcast_tuple = tuple(broadcast_tuple)
        dx = dx[broadcast_tuple]

        
        #Determine if we are reversing cumulative orientation
        if i in reverse_axes:
            #Set a slice that reversese only axis i in the PDF
            reverseSlice = rank*[slice(None,None,None)]
            reverseSlice[iprime] = slice(None,None,-1)
            reverseSlice = tuple(reverseSlice)
            #Do the cumulative calculation on the reversed PDF (and reverse back to the original orientation)
            cpdf = np.cumsum(cpdf[reverseSlice]*dx,axis=iprime)[reverseSlice]
        else:
            #Do the cumulative calculation
            cpdf = np.cumsum(cpdf*dx,axis=iprime)
        
    #Return the cumulative PDF
    return cpdf


def calculate_probability_contour(pdf,axes,pvals,axis=0):
    """ Calculates the probability level below which the PDF integrates to pval. 
    
        input:
        ------
            pdf  : a conditional PDF from fastKDE
            
            axes : a list of axis values returned from fastKDE
            
            pvals : the integrated amount of PDF that should be outside a contour of constant PDF (scalar or array-like)
            
        output:
        -------
        
            an array of PDF values for which the integral of the PDF outside the PDF value should integrate to pval.
            
            Each value in the array corresponds to a value within axes[axis] (nominally, axis should be the variable
            on which the PDF is conditioned).
            
    
    """
    
    # calculate the dx values
    diff_axes = []
    for i in range(len(axes)):
        if i != axis:
            dx = np.diff(axes[i])
            dx = np.concatenate((dx,[dx[-1]]))
            diff_axes.append(dx)
            
    # create a meshgrid of dx values
    dxmesh = np.meshgrid(*diff_axes)
    # create the dx product
    dxgrid = np.prod(dxmesh,axis=0)
    # broadcast the dxgrid to the shape of the PDF
    broadcast_shape = len(axes)*[slice(None,None,None)]
    broadcast_shape[axis] = np.newaxis
    dxgrid = dxgrid[tuple(broadcast_shape[::-1])]
    
    # multiply the pdf by dx
    pdf_dx = dxgrid*pdf
    
    # reshape the PDF to unravel the summation dimensions
    new_shape = [-1,len(axes[axis])]
    pdf_unraveled = np.reshape(pdf,new_shape)
    # get the indices to sort by probability
    isort_inds = np.ma.argsort(pdf_unraveled,axis=0)
    # generate an index array suitable for use in a numpy array
    isort = ([ np.array(i) for i in zip(*isort_inds.T)], np.arange(len(axes[axis]),dtype=int)[np.newaxis,:])
    # sort by probability
    pdf_dx_unraveled = np.reshape(pdf*dxgrid,new_shape)[isort]
    
    # take the cumulative sum
    pdf_csum = np.ma.cumsum(pdf_dx_unraveled,axis=0)
    

    # find the location along each axis nearest to the given PDF value
    ind_closest = np.ma.argmin(abs(pdf_csum-pvals),axis=0)
    ind_closest = (ind_closest, np.arange(len(axes[axis]),dtype=int))

    # get the probability value at that location
    pdf_vals = np.reshape(pdf,new_shape)[isort][ind_closest]

    # return the value
    return pdf_vals

def pair_plot(var_list, \
              conditional = False, \
              fig_size = None, \
              var_names = None, \
              draw_scatter = None, \
              axis_limits = None, \
              log_scale = False, \
              auto_show = True, \
              cdf_levels = [0.1,0.25,0.5,0.75,0.9], \
              axis_lengths = None):
    """ Generate a multivariate pair plot of the input data. 
    
        input:
        ------
        
            var_list     : a list of input variables (all must be vectors of the same length) with at least two variables
              
            conditional  : flags whether to plot the bivariate PDFs as conditionals.

            var_names    : a list of strings corresponding to variable names
              
            fig_size     : The size of the figure (if none, it is set to 10,10)
            
            draw_scatter : flag whether to draw scatter plots on top of the PDFs.  If None, this turns on/off
                           automatically, depending on the number of points (1000 is the upper cutoff)
            
            axis_limits  : limits for all variables (applies to all axes)

            auto_show    : flags whether to automatically call PP.show() (no values are returned if this flag is on)
            
            cdf_levels   : a set of CDF levels (betwen 0 and 1 exclusive) for which to plot PDF contours
                           (i.e., a value of 0.5 means that 50% of the PDF falls outside the corresponding contour
                           of constant PDF that will be plotted).  If None is given, matplotlib will choose PDF levels.

            log_scale    : flags whether to use the logAxes argument for fastKDE.  If a single bool value, it applies
                           to all variables; if a list, each item in the list corresponds to a variable
            
            axis_lengths : The length of axis variables. If axis_lengths is None, then fastKDE automatically sets
                           axis lengths
                           
            
        output:
        -------

            **NOTE**: no values are returned if auto_show is True
        
            fig, axs, marginal_vals, marginal_pdfs, bivariate_pdfs   
            
                fig, axs : a matplotlib figure and axis matrix object
                
                marginal_vals : the values at which the PDFs (both marginal and bivariate) are defined
                
                marginal_pdfs : the marginal PDFs of the input variables
                
                bivariate_pdfs : the bivariate PDFs
    """
    
    # convert the list to an array
    try:
        input_var_array = np.array(var_list)
    except:
        raise ValueError("var_list could not be converted to a numpy array; are all vectors of the same length?")
        
    # get the array rank 
    rank = len(input_var_array.shape)
    if rank != 2:
        raise ValueError("input variables in var_list must be vectors")
    
    
    # get the array shape
    num_vars = input_var_array.shape[0]
    if num_vars < 2:
        raise ValueError("pair_plot requires at least two variables in var_list")
    
    if log_scale is False or log_scale is True:
        log_scale = num_vars*[log_scale]
   
    marginal_pdfs = num_vars*[None]
    marginal_vals = num_vars*[None]
    var_limits = num_vars*[None]

    if draw_scatter is None:
        if input_var_array.shape[1] > 1000:
            draw_scatter = False
        else:
            draw_scatter = True
    
    # calculate the marginals and define limits
    for n in range(num_vars):
        # calculate PDFs
        marginal_pdfs[n], marginal_vals[n] = fastKDE.pdf(input_var_array[n,:],logAxes = log_scale[n])
        
        # define axis limits
        if axis_limits is None:
            var_limits[n] = [input_var_array[n,:].min(),input_var_array[n,:].max()]
        else:
            var_limits[n] = axis_limits
        
        
    #bivariate_pdfs = num_vars*[num_vars*[None]]
    bivariate_pdfs = {}
    
    # calculate the bivariate PDFs
    for n1 in range(num_vars):
        for n2 in range(n1, num_vars):
            if n1 != n2:
                if not conditional:
                    bivariate_pdfs[n1,n2], _ = fastKDE.pdf(input_var_array[n1,:], input_var_array[n2,:],logAxes = [log_scale[n1],log_scale[n2]])
                else:
                    bivariate_pdfs[n1,n2], _ = fastKDE.conditional(input_var_array[n2,:], input_var_array[n1,:],logAxes = [log_scale[n1],log_scale[n2]])
                    bivariate_pdfs[n2,n1], _ = fastKDE.conditional(input_var_array[n1,:], input_var_array[n2,:],logAxes = [log_scale[n2],log_scale[n1]])

            
            
    # set variable labels if needed
    if var_names is None:
        var_names = ['var {}'.format(n) for n in range(num_vars)]
        
    # set the figure size if needed
    if fig_size is None:
        fig_size = (10,10)
        
    # create the figure
    fig, axs = PP.subplots(num_vars,num_vars,figsize=fig_size)
    
    # plot the marginals
    for n in range(num_vars):
        axs[n,n].plot(marginal_vals[n],marginal_pdfs[n])
        axs[n,n].set_xlim(var_limits[n])
        axs[n,n].tick_params(labelleft = False)    
        if n < num_vars - 1:
            axs[n,n].tick_params(labelbottom = False)    
        else:
            axs[n,n].set_xlabel(var_names[n])
            
        if log_scale[n]:
            axs[n,n].set_xscale('log')
        
            
    # plot the bivariate PDFs
    for n1 in range(num_vars):
        for n2 in range(n1, num_vars):
            if n1 != n2:
                
                if not conditional:
                    # get the PDF levels
                    if cdf_levels is None:
                        levels = None
                    else:
                        levels = np.sort(np.array([calculate_probability_contour(  bivariate_pdfs[n1,n2][...,np.newaxis],\
                                                                           [[0],marginal_vals[n1],marginal_vals[n2]],\
                                                                           c) 
                                                   for c in cdf_levels])).squeeze()
                    
                    # plot the PDF
                    pdf_to_plot = bivariate_pdfs[n1,n2]
                    if any([log_scale[n1],log_scale[n2]]):
                        pdf_to_plot = np.ma.log(np.ma.masked_less_equal(bivariate_pdfs[n1,n2],0))
                        levels = np.log(levels)
                else:
                    levels = cdf_levels
                    pdf_to_plot = cumulative_integral(bivariate_pdfs[n1,n2],[marginal_vals[n1],marginal_vals[n2]],integration_axes=1)
                    # mask CDF parts that don't normalize to something close to 1
                    pdf_to_plot = np.ma.masked_where(np.logical_not(np.isclose(np.ma.ones(pdf_to_plot.shape)*pdf_to_plot[-1,:][np.newaxis,:],1.0)),pdf_to_plot)

                
                axs[n2,n1].contour(marginal_vals[n1],marginal_vals[n2],pdf_to_plot,levels=levels)
                    
                if log_scale[n1]:
                    axs[n2,n1].set_xscale('log')
                if log_scale[n2]:
                    axs[n2,n1].set_yscale('log')

                # plot the scatter
                if draw_scatter:
                    axs[n2,n1].plot(input_var_array[n1,:],input_var_array[n2,:],'k.',alpha = 0.3)
                    
                # set axis limits
                axs[n2,n1].set_xlim(var_limits[n1])
                axs[n2,n1].set_ylim(var_limits[n2])

                if not conditional:
                    # turn of axes for the other part of the triangle
                    axs[n1,n2].axis('off')
                else:

                    levels = cdf_levels
                    pdf_to_plot = cumulative_integral(bivariate_pdfs[n2,n1],[marginal_vals[n2],marginal_vals[n1]],integration_axes=1)
                    # mask CDF parts that don't normalize to something close to 1
                    pdf_to_plot = np.ma.masked_where(np.logical_not(np.isclose(np.ma.ones(pdf_to_plot.shape)*pdf_to_plot[-1,:][np.newaxis,:],1.0)),pdf_to_plot)

                    axs[n1,n2].contour(marginal_vals[n2],marginal_vals[n1],pdf_to_plot,levels=levels)
                        
                    if log_scale[n2]:
                        axs[n1,n2].set_xscale('log')
                    if log_scale[n1]:
                        axs[n1,n2].set_yscale('log')

                    # plot the scatter
                    if draw_scatter:
                        axs[n1,n2].plot(input_var_array[n2,:],input_var_array[n1,:],'k.',alpha = 0.3)
                        
                    # set axis limits
                    axs[n1,n2].set_xlim(var_limits[n2])
                    axs[n1,n2].set_ylim(var_limits[n1])


        
        
    
    # turn off axis limits and set axis labels
    for n1 in range(num_vars):
        for n2 in range(n1, num_vars):
            if n1 != n2:
                if n1 == 0: 
                    axs[n2,n1].set_ylabel(var_names[n2])
                    axs[n2,n1].tick_params(labelleft = True)    
                else:
                    axs[n2,n1].tick_params(labelleft = False)    
                    
                if n2 == num_vars - 1:
                    axs[n2,n1].set_xlabel(var_names[n1])
                    axs[n2,n1].tick_params(labelbottom = True)    
                else:
                    axs[n2,n1].tick_params(labelbottom = False)    

                if conditional:
                    axs[n1,n2].tick_params(labelleft = False)    
                    axs[n1,n2].tick_params(labelbottom = False)    
                    if n2 == num_vars - 1:
                        axs[n1,n2].tick_params(labelright = True)    
                    
    if auto_show:
        PP.show()
        return
    
    return fig, axs, marginal_vals, marginal_pdfs, bivariate_pdfs



