# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:41:20 2014

@author: ondrej
"""

import glob, pylab, plotting
import numpy as np
import matplotlib.cm as cm


#def plotR(filename = "Results/R_*",showLegend = True, showNumbers = True, plotGraph = True, fontsize = 5):
#    pylab.figure()
#    leg = []
#    Rmin=1.
#    index = 0
#    files = glob.glob(filename)
#    rall = np.zeros(len(files))
#    nfiles = len(files)
#    colors = cm.rainbow(np.linspace(0, 1, nfiles))
#    for filename, c in zip(files, colors):
#        R = np.load(filename)
#        pylab.semilogy(R,c=c)
#        if showNumbers: 
#            pylab.text(len(R),R[-1],str(index))
#    #    pylab.plot(np.load(filename))
#        leg.append(str(index) + ' ' + filename)
#        rall[index] = R[-1]
#        index += 1    
#        if R[-1]<Rmin:
#            evMin=filename
#            Rmin=R[-1]
#
#    pylab.grid(b=True, which = 'both')    
#    if showLegend:
#        pylab.legend(leg, fontsize = fontsize)
#    print "Best: " + evMin + " @ " + str(Rmin)
#    return leg, evMin
#    
def plotR(filename = "Results/R_*",showLegend = True, showNumbers = True, plotGraph = True, fontsize = 5):
    pylab.figure()
    leg = []
    Rmin=1.
    index = 0
    files = glob.glob(filename)
    rall = np.zeros(len(files))
    nfiles = len(files)
    colors = cm.rainbow(np.linspace(0, 1, nfiles))
    for filename in files:
        R = np.load(filename)
        rall[index] = R[-1]        
        index += 1  
    
    ind_sort = [i[0] for i in sorted(enumerate(rall), key=lambda x:x[1])]
    files_sort =  list( files[i] for i in ind_sort)
    rall_sort =  list( rall[i] for i in ind_sort)
    index = 0
    for filename, c in zip(files_sort, colors):
        R = np.load(filename)
        pylab.semilogy(R,c=c)
        if showNumbers: 
            pylab.text(len(R),R[-1],str(index))
        leg.append(str(index) + ' ' + filename)
        index += 1

    pylab.grid(b=True, which = 'both')    
    if showLegend:
        pylab.legend(leg, fontsize = fontsize)
        
    evMin = files_sort[0]
    Rmin = rall_sort[0]
    print "Best: " + evMin + " @ " + str(Rmin)
    return leg, evMin, rall_sort
    
def getIms(filename = "Results/obj_*"):
    
    files = glob.glob(filename)
    im_all = np.load(files[0])[np.newaxis]
    for filename in files[1:]:
        im = np.load(filename)
        im_all = np.concatenate((im_all, im[np.newaxis]),0)        
    return im_all, files
    