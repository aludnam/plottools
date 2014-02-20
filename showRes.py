# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:01:39 2013

@author: ondrej
"""
import pylab
import plotting as ptg

def plotCplx(im):
    """
    Shows modulus an phase of the complex image. 
    """
    
    modulus = abs(im)
    phase = pylab.angle(im)    
    pylab.figure(figsize=(8,4));

    pylab.subplot(121)
    pylab.imshow(modulus)    
    pylab.xlabel('pixels')
    pylab.ylabel('pixels')
    pylab.grid(True)
    
    pylab.subplot(122)
    pylab.imshow(phase)    
    pylab.xlabel('pixels')
    pylab.ylabel('pixels')
    pylab.grid(True)
    
    pylab.axes((0.84,0.45,0.1,.1), axisbg='w') #[left, bottom, width, height]
    ptg.colourwheel() 
    pylab.tight_layout()      
