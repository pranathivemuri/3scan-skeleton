# 3D Image Skeletonization Tools

A lot of these tools are written over 2015 - 2017 mostly when I was a part of 3Scan as an intern. Thanks to Matthew Goodman, Mike Pesavento, Jenny Folkesson, Ted Blackman, and Joanna Lipinski for their collaborations and ideas.

This repository contains programs needed to obtain a 3D skeleton and quantify the skeletonized array to statistics.
Input must be a binary array with z in its first dimension.  
Skeletonization on a 3D binary array is performed by iteratively removing the boundary points until a line in the center is obtained by convolving the image with structuring elements from the [paper](https://drive.google.com/file/d/1kCEmfOx1mwoyggAfkYOsyRhOywfkiIU1/view?usp=sharing).  
This function is implemented using [cython](http://docs.cython.org/src/reference/compilation.html) for fast execution and pyximport is used to automatically build and use the function

Using mayavi is a useful way to visualize these test stacks  
Install mayavi via:  

```conda install -c menpo mayavi=4.5.0```

This version works on python 3  
Versions < 4.5 don't work on python 3.x  
mayavi also requires that QT4 (pyqt) is installed  
This may be in conflict with matplotlib >=1.5.3, which started to use qt5. So, use mayavi and matplotlib in a separate [conda](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04) environment to avoid conflicts

View the stack contours:

```import mayavi as mlab```  
```mlab.contour3d(anynpynonbooleanarray)```  
```mlab.options.offscreen = True```  
```mlab.savefig("arrayName.png")```  

To validate two networks use [NetMets](http://stim.ee.uh.edu/resources/software/netmets/) - Software wrote and maintained under UH STIM Lab by Dr.Mayerich, I wrote a part of it while I was pursuing my masters at UH.

[Github code](https://git.stim.ee.uh.edu/segmentation/netmets)  
[Build guide](http://stim.ee.uh.edu/education/software-build-guide/)  
After installing NetMets, validate skeleton generated after converting into obj filing and comparing with ground truth using

```netmets objfile1 objfile2 --sigma 3```