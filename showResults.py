# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:01:39 2013

@author: ondrej
"""
import numpy as np
import plotting as ptg
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def showIm(im,tit=None,name=None,pixSizeDirect_um=None):
    if pixSizeDirect_um==None:
        pixSizeDirect_um=1
        units='[pix]'
    else:
        units='[um]'
        
    plt.figure()
    plt.imshow(im,extent=(0,im.shape[1]*pixSizeDirect_um,0,im.shape[0]*pixSizeDirect_um),interpolation='nearest')
    plt.grid(True,color='white')
    plt.xlabel('x '+units)
    plt.ylabel('y '+units)
    if not(tit==None):
        plt.title(tit)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(cax=cax)                      
    if not(name==None):
        plt.savefig(name)

def ima(img,**kwargs):
    plt.figure()
    print "Shown in figure %g."%plt.get_fignums()[-1]
    ptg.im(abs(img),aspect='equal')
    
def imp(img,**kwargs):
    plt.figure()
    print "Shown in figure %g."%plt.get_fignums()[-1]
    ptg.im(np.angle(img),aspect='equal',cmap='hsv',vmin=-np.pi,vmax=np.pi)    
    ptg.insertColorwheel()
    
def imap(img):
    plt.figure(figsize=(8, 4))
    print "Shown in figure %g."%plt.get_fignums()[-1]
    plt.subplot(1,2,1)
    plt.imshow(abs(img),aspect='equal')
    plt.title('amplitude')
    plt.subplot(1,2,2)
    plt.imshow(np.angle(img),aspect='equal',cmap='hsv',vmin=-np.pi,vmax=np.pi)
    plt.title('phase')
    ptg.insertColorwheel(left=.8, bottom=.15)