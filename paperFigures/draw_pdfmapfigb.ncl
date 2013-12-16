load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$HOME/src/climes_ncl/earthPhysics.ncl"
load "./markerSpec.ncl" ;Read in plot marker locations

begin

  rmax = 500. * 1000. ;Set the maximum size used in the exponent fit

;*******************************************************************************
;*******************************************************************************
;********************** Read in and prepare data *******************************
;*******************************************************************************
;*******************************************************************************

  ;Open the file that contains postprocessed information about the PDF
  infilename = "./postHocAnalysis2.nc"
  fin = addfile(infilename,"r")

  ;Read in the distribution moments
  m2 = fin->structureFunctions(:,:,1,0:0) ;Read in the second moment from distance index 3 to the end
  m4 = fin->structureFunctions(:,:,3,0:0) ;Read in the fourth moment from distance index 3 to the end

  ;Set the fillValues
  m2@_FillValue = -9999.
  m4@_FillValue = -9999.

  ;Read the dist/lat/lev variables
  dist = m2&dist
  lat = m2&lat
  lev = m2&lev
  nlat = dimsizes(lat) ;Get the number of latitudes

  ;Initialize some dummy variables to the correct size
  incDists = m2 ;Increment distances
  latC = m2 ;Latitudes
  dX = m2 ;Grid spacings

  ;Create size-conformant versions of the lat and dist dimensions
  latC = conform_dims(dimsizes(m2),lat,1)
  distC = conform_dims(dimsizes(m2),dist,2)

  ;Calculate the grid spacing at the equator
  dXEquator = rearth*degtorad*(lat(nlat/2) - lat(nlat/2-1))
  ;Calculate the grid spacing at each latitude
  dX = abs(dXEquator*cos(degtorad*latC))
  ;Calculate the increment spacings at each latitude
  incDists = dX*distC

  ;Remove values of m2 and m4 where the increment distance
  ;is less than rmax
  m2 = where(incDists.le.rmax,m2,m2@_FillValue)
  m4 = where(incDists.le.rmax,m4,m4@_FillValue)

  ;Calculate the average second and fourth moment over the chosen distribution subset
  ;(this is okay because the distributions are self-similar over the chosen range)
  m2bar = dim_avg(m2)
  m4bar = dim_avg(m4)

  exKurt = m2(:,:,0) ;initialize the excess kurtosis variable
  m2bar = where(m2bar.gt.0,m2bar,m2bar@_FillValue) ; Remove any cells with a second moment of 0 (or less), which signals bad data
  kurt = m4bar/m2bar^2 ;Calculate the kurtosis
  exKurt = kurt - 3. ;Remove 3 to calculate the excess kurtosis

;*******************************************************************************
;*******************************************************************************
;********************** Set plot settings **************************************
;*******************************************************************************
;*******************************************************************************
  res = True

  ;Set contour drawing properties
  res@cnFillMode = "RasterFill" ;Don't smoothe the data
  res@cnFillOn = True ;Color-fill the raster boxes
  res@cnLinesOn = False  ;Don't draw contour lines
  res@cnLineLabelsOn = False ;Don't draw contour line labels
  myColors = read_colormap_file("StepSeq25") ;Read a color map file
  res@cnFillPalette = myColors(4:,:) ;Set the colormap (and offset it slightly
                                     ;so that 2 sits at a color boundary)
  ;Set axis properties (bounds, etc.)
  res@trXMinF = -50. ;Set the plot's left bound
  res@trXMaxF = 50. ;Set the plot's right bound
  res@trYMaxF = 1050. ;Set the plot's lower bound
  res@trYMinF = 80. ;Set the plot's upper bound
  res@trYReverse = True ;Put higher pressures at the bottom

  ;Set axis labels
  res@tiYAxisString = "Pressure (hPa)"

  ;Set the title (variable name)
  exKurt@long_name = "Excess kurtosis,~F33~g~F21~~B~2~N~"
  exKurt@units = ""

  ;Set attributes of the plot's geometry
  res@vpWidthF = 0.7 ;Set the (relative) physical width of the plot
  res@vpHeightF = 0.3 ;Set the (relative) physical width of the plot
;  res@gsnMaximize = True ;Tell NCL to remove as much whitespace as possible

  ;Set the countour levels
  res@cnLevelSelectionMode = "ExplicitLevels"
 ; res@cnLevels = ispan(-250,800,25)/100.
  res@cnLevels = ispan(-350,1000,25)/100.

  ;Tell NCL not to draw or advance the frame (so we can add text/annotation)
  res@gsnFrame = False
  res@gsnDraw = False

;*******************************************************************************
;*******************************************************************************
;********************** Draw the plot and annotations **************************
;*******************************************************************************
;*******************************************************************************

  ;Create the plot
  wks = gsn_open_wks("eps","pdfmapfigb") ;Open the worksheet (file)
  plot = gsn_csm_contour(wks,exKurt,res) ;Create the plot

  ;Draw the plot
  draw(plot)

  ;Draw the subfigure label
  resText = True
  resText@txFontHeightF = 0.025
  gsn_text(wks,plot,"(b)",-46.,100.,resText)

  ;Draw the location markers specified in markerSpec.ncl
  resMark = True
  resMark@txFontHeightF = 0.015 ;Set a smaller fontheight
  do n = 0,nmarkers-1
    gsn_text(wks,plot,""+(n+1),lat(iLatMarker(n)),lev(iLevMarker(n)),resMark)
  end do

  ;Write the plot to disk
  frame(wks)


end
