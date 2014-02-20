# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:41:20 2014

@author: ondrej
"""

import glob, pylab
import numpy as np
import matplotlib.cm as cm


def plot(filename = "Results/R_*",showLegend = True, showNumbers = True, plotGraph = True, fontsize = 5):
    pylab.figure()
    leg = []
    Rmin=1.
    index = 0
    files = glob.glob(filename)
    nfiles = len(files)
    colors = cm.rainbow(np.linspace(0, 1, nfiles))
    for filename, c in zip(files, colors):
        R = np.load(filename)
        pylab.semilogy(R,c=c)
        if showNumbers: 
            pylab.text(len(R),R[-1],str(index))
    #    pylab.plot(np.load(filename))
        leg.append(str(index) + ' ' + filename)
        index += 1    
        if R[-1]<Rmin:
            evMin=filename
            Rmin=R[-1]

    pylab.grid(b=True, which = 'both')    
    if showLegend:
        pylab.legend(leg, fontsize = fontsize)
    print "Best: " + evMin + " @ " + str(Rmin)
    return leg, evMin