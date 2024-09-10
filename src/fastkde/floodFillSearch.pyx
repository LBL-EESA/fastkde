# cython: language_level=2

import numpy as np
cimport numpy as np
cimport cython

cdef inline np.int64_t ravel_shift(   tuple indices, \
                                np.int64_t array_rank, \
                                np.int64_t [:] array_shape, \
                                np.int64_t dimension,  \
                                np.int64_t amount,     \
                                np.int64_t dimension_wraps):
    """Return the raveled index of a shifted version of indices, where a
    specific dimension has been shifted by a certain amount.  If wrapping is
    not flagged and the shift is out of bounds, returns -1"""


    cdef np.int64_t running_product
    cdef np.int64_t n
    cdef np.int64_t i
    cdef np.int64_t npp
    cdef np.int64_t this_index

    running_product = 1
    i = 0

    #Loop over dimensions, starting at the rightmost dimension
    for n in range(array_rank,0,-1):
        #Calculate the running product of dimension sizes
        if( n != array_rank):
            running_product *= array_shape[n]

        #Set the current index
        this_index = indices[n-1]

        npp = n-1
        if(npp == dimension):
            #If this is the shifting dimension,
            #increment it
            this_index += amount

            #Check if we need to deal with a
            #wrap around dimension
            if(dimension_wraps):
                if(this_index < 0):
                    this_index += array_shape[npp]
                if(this_index >= array_shape[npp]):
                    this_index -= array_shape[npp]

            #Check if the current index is out of bounds;
            #return -1 if so
            if(this_index < 0 or this_index >= array_shape[npp]):
                i = -1
                break

        #increment the counter
        i += running_product*this_index

    #Check whether the index is within the memory bounds of the array
    #return the -1 flag if not
    running_product *= array_shape[0]
    if(i >= running_product or i < 0):
        i = -1

    return i

@cython.boundscheck(False)
cdef tuple findNeighbors(   np.int64_t raveled_start_index, \
                            np.float_t search_threshold, \
                            np.int64_t [:] array_shape, \
                            np.int64_t array_rank, \
                            list dimension_wraps, \
                            np.float_t [:] input_array, \
                            np.int64_t [:] is_not_searched, \
                   ):
    """Does a flood fill algorithim on input_array in the vicinity of
    raveled_start_index to find contiguous areas where raveled_start_index > search_threshold 
    
        input:
        ------
            raveled_start_index   :   (integer) the index of input_array.ravel() at which to start

            search_threshold :   The threshold for defining fill regions
                                (input_array > search_threshold)

        output:
        -------
            A list of N-d array indices.
    
    """
    cdef list items_to_search #Running item search list
    cdef list contiguous_indices #A list of indices
    cdef np.int64_t r # current array dimension
    cdef np.int64_t test_index_left # A test index
    cdef np.int64_t test_index_right # A test index
    cdef tuple item_tuple # the tuple index of the current search item
    cdef np.int64_t shift_amount # indicates the shift direction when searching

    #Initialize the contiguous index list
    contiguous_indices = []

    #Initialize the search list
    items_to_search = [raveled_start_index]

    while items_to_search != []:

        #Get the index of the current item
        item_tuple = np.unravel_index(items_to_search[0],array_shape)

        for r in range(array_rank):
            #Shift the current coordinate to the right by 1 in the r dimension
            shift_amount = 1
            test_index_right = ravel_shift( \
                                        item_tuple, \
                                        array_rank, \
                                        array_shape, \
                                        r, \
                                        shift_amount,
                                        dimension_wraps[r])

            #Check that this coordinate is still within bounds
            if(test_index_right >= 0):
                #Check if this index satisfies the search condition
                if(input_array[test_index_right] > search_threshold and \
                        is_not_searched[test_index_right] == 1):
                    #Append it to the search list if so
                    items_to_search.append(test_index_right)
                    #Flags that this cell has been searched
                    is_not_searched[test_index_right] = 0


            #Shift the current coordinate to the right by 1 in the r dimension
            shift_amount = -1
            test_index_left = ravel_shift( \
                                        item_tuple, \
                                        array_rank, \
                                        array_shape, \
                                        r, \
                                        shift_amount,
                                        dimension_wraps[r])

            #Check that this coordinate is still within bounds
            if(test_index_left > 0):
                #Check if this index satisfies the search condition
                if(input_array[test_index_left] > search_threshold and \
                        is_not_searched[test_index_left] == 1 ):
                    #Append it to the search list if so
                    items_to_search.append(test_index_left)
                    #Flags that this cell has been searched
                    is_not_searched[test_index_left] = 0

 

        #Flag that this index has been searched
        #is_not_searched[tuple(items_to_search[0])] = 0
        #Now that the neighbors of the first item in the list have been tested,
        #remove it from the list and put it in the list of contiguous values
        contiguous_indices.append(items_to_search.pop(0))

    #Return the list of contiguous indices (converted to index tuples)
    return np.unravel_index(contiguous_indices,array_shape)

cpdef list flood_fill_search( \
                np.ndarray input_array, \
                np.float_t search_threshold = 0.0, \
                wrap_dimensions = None):
    """Given an N-dimensional array, find contiguous areas of the array
    satisfiying a given condition and return a list of contiguous indices
    for each contiguous area.
        
        input:
        ------

            input_array      :   (array-like) an array from which to search
                                contiguous areas

            search_threshold :   The threshold for defining fill regions
                                (input_array > search_threshold)

            wrap_dimensions :    A list of dimensions in which searching
                                should have a wraparound condition

        output:
        -------

            An unordered list, where each item corresponds to a unique
            contiguous area for which input_array > search_threshold, and
            where the contents of each item are a list of array indicies
            that access the elements of the array for a given contiguous
            area.

    """
    cdef np.ndarray[np.int64_t,ndim=1] array_shape
    cdef np.int64_t array_rank
    cdef np.int64_t num_array_elements
    cdef list dimension_wraps
    cdef list contiguous_areas


    #Determine the rank of input_array
    try:
        array_shape = np.array(np.shape(input_array),dtype=np.int64)
        array_rank = len(array_shape)
        num_array_elements = np.prod(array_shape)
    except BaseException as e:
        raise ValueError("input_array does not appear to be array like.  Error was: {}".format(e))

    #Set the dimension wrapping array
    dimension_wraps = array_rank*[False]
    if wrap_dimensions is not None:
        try:
            dimension_wraps[list(wrap_dimensions)] = True
        except BaseException as e:
            raise ValueError("wrap_dimensions must be a list of valid dimensions for input_array. Original error was: {}".format(e))

    #Set an array of the same size indicating which elements have been set
    cdef np.ndarray is_not_searched
    is_not_searched = np.ones(array_shape,dtype = 'int')

    #Set the raveled input array
    cdef np.float_t [:] raveledInputArray = input_array.ravel()
    #And ravel the search inidcator array
    cdef np.int64_t [:] raveledIsNotSearched = is_not_searched.ravel()
    
    #Set the search list to null
    contiguous_areas = []

    cdef np.int64_t i
    #Loop over the array
    for i in range(num_array_elements):
        #print "{}/{}".format(i,num_array_elements)
        #Check if the current element meets the search condition
        if raveledInputArray[i] >= search_threshold and raveledIsNotSearched[i]:
            #Flag that this cell has been searched
            raveledIsNotSearched[i] = 0

            #If it does, use a flood fill search to find the contiguous area surrouinding
            #the element for which the search condition is satisified. At very least, the index
            #of this element is appended to contiguous_areas
            contiguous_areas.append(\
                                    findNeighbors(  i,  \
                                                    search_threshold,    \
                                                    array_shape,         \
                                                    array_rank,          \
                                                    dimension_wraps,     \
                                                    raveledInputArray,         \
                                                    raveledIsNotSearched      ))

        else:
            #Flag that this cell has been searched
            raveledIsNotSearched[i] = 0
                                    


    #Set the list of contiguous area indices
    return contiguous_areas

def sort_by_distance_from_center(inds,var_shape):
    """Takes sets of indicies [e.g., from flood_fill_searchC.flood_fill_search()] and sorts them by distance from the center of the array from which the indices were taken.
    
        input:
        ------
        
            inds     :  a list of tuples of numpy ndarrays (of type integer and
                        rank 1), where each tuple item contains a vector of
                        indices for each index of an array.  Each list item
                        should conform to the output of the numpy where()
                        function.  It is assumed that each set of indices
                        represents a contiguous portion of an array.
                       
            var_shape : the shape of the variable from which inds originate
            
        returns:
        --------

             A sorted version of inds, where the items are sorted by the
             distance of the contiguous area relative to the center of the
             array whose shape is var_shape.  The first item is the closest to
             the center of the array.
             
    """
    #Get the center index
    center = np.around(np.array(var_shape)/2)
    
    #Transform the indices to be center-relative
    centeredInds = [ tuple([ aind - cind] for aind,cind in zip(indTuples,center)) for indTuples in inds ]
    
    #Calculate center-of-mass ffor each contiguous array
    centersOfMass = [ np.array([np.average(aind) for aind in indTuples]) for indTuples in centeredInds]
    
    #Calculate the distance from the origin of each center of mass
    distances = [ np.sqrt(sum(indices**2)) for indices in centersOfMass]
    
    #Determine the sorting indices that will sort inds by distance from the center
    isort = list(np.argsort(distances))
    
    #Return the sorted index array
    return [inds[i] for i in isort]
