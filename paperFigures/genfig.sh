#!/bin/bash

printusage()
{
  echo "Usage: $0 file1.tex [file2.tex ... ]"
}

if [ "$#" -lt "1" ]; then
  printusage
  exit
fi

for myfile in "$@"
do
  mysuffix=${myfile##*.}

  if [ "$mysuffix" != "tex" ]; then
    echo "Error: $myfile does not end in .tex"
    echo ""
    printusage
    exit
  fi

  #Run latex on the latex source file
  latex $myfile

  myfiledvi=${myfile/tex/dvi}
  myfileeps=${myfile/tex/eps}

  #Convert the resulting DVI file to an encapsulated postscript file
  dvips -E ${myfiledvi} -o ${myfileeps}
  epspdf ${myfileeps}

  #Uncomment the next line to convert this postscript file to a jpeg
  #(requires ImageMagick to be installed)
  #convert -density 300x300 rtsinglediff.eps rtsinglediff.jpg

done
