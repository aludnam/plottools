# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:37:26 2013

# to do: statistic for var, max and sum frames
@author: ondrej
"""
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.fft import fft2,ifft2, fftshift, ifftshift
import scipy.ndimage as nd
import pylab

#pylab.matplotlib.use('Qt4Agg') #set the right backend

class ims:
    """
    plotting.ims(image)

    Shows individual frames in the 3D image (dimensions organized as [z,x,y]).
    "n" next frame, ("N" next by step of 10)
    "p" previous frame, ("P" previous by step of 10)
    "l" toogles the log scale.
    "c" swithces between 'grey','jet','hot' and 'hsv' colormaps.
    "i" zooms in
    "o" zooms out
    "arrows" move the frame around
    "r" resets the position to the center of the frame
    "g" toogle local/global stretch
    "q/Q" increase / decrease the lower limit of the contrast
    "w/W" increase / decrease the upper limit of the contrast
    "R" reset contrast to default
    "S" shows sum projection
    "M" shows max projection
    "V" shows var projection
    "T" prints statistics

    Works with qt backend -> start ipython as: "ipyton --matplotlib qt" or do e.g.: "matplotlib.use('Qtg4Agg')"

    The core of the class taken from http://stackoverflow.com/questions/6620979/2d-slice-series-of-3d-array-in-numpy
    """
    pylab.rcParams['keymap.yscale'] = '' # to disable the binding of the key 'l'
    pylab.rcParams['keymap.pan'] = '' # to disable the binding of the key 'p'
    pylab.rcParams['keymap.grid'] = '' # to disable the binding of the key 'g'
    pylab.rcParams['keymap.zoom'] = '' # to disable the binding of the key 'o'
    def __init__(self, im, i=0, titles=None):
        pylab.ion()
        self.dtype = im.dtype
        if im.ndim is 2:
            #if pylab.iscomplex(im).any():
            if isinstance(im.flatten()[0],(complex,np.complexfloating)):
                self.complex = True
                self.im = pylab.dstack([abs(im),pylab.angle(im)])
            else:
                self.complex = False
                self.im = im
        if im.ndim is 3:
            im = abs(im)
            self.complex = False
            self.im = np.dstack(im) # first dimension is the index of the stack
#            if im.shape[0]<im.shape[2]:
#                self.im = np.dstack(im)
#            else:
#                self.im = im
        self.i = i
        self.titles = titles
        self.logon = 0
        self.cmap = 0
        self.projToggle = 0
        self.zoom = 1
        self.globalScale = 0
        self.offx,self.offy = 0,0
        self.stepXY = 10 # step of the movement up-down, left-righ
        self.vmin,self.vmax = abs(im).min(),abs(im).max()
        fim = np.log10(self.im.flatten())
        if all(fim==-np.inf): # this is for the zero image
            self.vminLog,self.vmaxLog=-np.inf,-np.inf
        elif all(fim==np.inf): # this is for the inf image
            self.vminLog,self.vmaxLog=np.inf,np.inf
        elif all(np.isnan(fim)): # this is for all negative
            self.vminLog,self.vmaxLog=0,0
        else:
            self.vminLog,self.vmaxLog = fim[fim>-np.inf].min(),fim[fim<np.inf].max()
#        if self.vminLog == self.vmaxLog:
#            self.vmaxLog += sys.float_info.epsilon
        self.offVmin,self.offVmax = 0,0
        self.frameShape = self.im.shape[:2]
        self.showProfiles = False
        if not(self.showProfiles):
            self.fig = pylab.figure()
            self.figNum = pylab.get_fignums()[-1]
            print("Shown in figure %g."%self.figNum)
            self.ax = self.fig.add_subplot(111)
        else:
            ################
            # definitions for the axes
            widthProf = 0.1
            left, width = 0.05, 0.75
            bottomProf = 0.05
            bottom, height = widthProf + bottomProf + 0.05, 0.75

            leftProf = left + width + 0.05

            rect_im = [left, bottom, width, height]
            rect_X = [left, bottomProf, width, widthProf] # horizontal
            rect_Y = [leftProf, bottom, widthProf, height] # vertical

            # start with a rectangular Figure
            self.fig = pylab.figure(figsize=(8,8))
            self.ax = plt.axes(rect_im)
            self.axX = plt.axes(rect_X)
            self.axY = plt.axes(rect_Y)

            nullfmt = pylab.NullFormatter()         # no labels
            self.axX.xaxis.set_major_formatter(nullfmt)
            self.axX.yaxis.set_major_formatter(nullfmt)
            self.axY.xaxis.set_major_formatter(nullfmt)
            self.axY.yaxis.set_major_formatter(nullfmt)
            self.posProfHoriz = round(self.frameShape[0]/2)
            self.posProfVert = round(self.frameShape[1]/2)

        ################

        self.draw()
        self.fig.canvas.mpl_connect('key_press_event',self)

    def draw(self):
        pylab.cla()
        tit=str()
        if self.im.ndim is 2:
            im = self.im
        if self.im.ndim is 3:
            im = self.im[...,self.i]
            tit='frame {0}'.format(self.i)+'/'+str(self.im.shape[2]-1)
            # projections
            if self.projToggle:
                if self.projType=='M':
                    im=self.im.max(2)
                    tit = 'max projection'
                if self.projType=='S':
                    im=self.im.sum(2)
                    tit = 'sum projection'
                if self.projType=='V':
                    im=pylab.var(self.im,2)
                    tit = 'var projection'
        if self.complex:
            tit += ', cplx (0=abs,1=phase)'
        if self.logon:
            tit += ', log10'
            minval = 0 #sys.float_info.epsilon
            if (im<minval).any():
                im2show = np.log10(im.clip(min=minval))
            else:
                im2show = np.log10(im)
            im2show = np.log10(im)
            if self.globalScale:
                vrange = self.vmaxLog - self.vminLog
                vmin_tmp=self.vminLog + self.offVmin*vrange
                vmax_tmp=self.vmaxLog - self.offVmax*vrange
                tit += ', global scale'
            else:
                fi = im2show.flatten()
                immin = fi[fi>-np.inf].min()
                immax = fi[fi<np.inf].max()
                vrange = immax - immin
                vmin_tmp = immin + self.offVmin*vrange
                vmax_tmp = immax - self.offVmax*vrange
        else:
            tit += ', lin'
            im2show = im
            if self.globalScale:
                vrange = self.vmax-self.vmin
                vmin_tmp=self.vmin + self.offVmin*(vrange)
                vmax_tmp=self.vmax - self.offVmax*(vrange)
                tit += ', global scale'
            else:
                immin,immax = im2show.min(),im2show.max()
                vrange = immax - immin
                vmin_tmp = immin + self.offVmin*vrange
                vmax_tmp = immax - self.offVmax*vrange

        if self.zoom > 1:
            tit += ', zoom %g x'%(self.zoom)

        if self.offVmin or self.offVmax:
            tit += ', contrast L%g %% H%g %%'%(round(self.offVmin*100),round(self.offVmax*100))
        if self.cmap==0:
            pylab.gray()
        elif self.cmap==1:
            pylab.jet()
        elif self.cmap==2:
            pylab.hot()
        elif self.cmap==3:
            pylab.hsv()
        if self.titles==None:
            self.ax.set_title(tit)
        else:
            self.ax.set_title(self.titles[self.i])
        pylab.show()
        hx,hy = self.frameShape[0]/2., self.frameShape[1]/2.
        lx,ly = hx/(self.zoom),hy/(self.zoom)

        rx_low = max(min(np.floor(hx) + self.offx - np.floor(lx),self.frameShape[0]-self.stepXY),0)
        rx_high = min(max(np.floor(hx) + self.offx + np.ceil(lx),self.stepXY),self.frameShape[0])
        rx = np.arange(rx_low,rx_high)[:,np.newaxis].astype(int)


        ry_low = max(min(np.floor(hy) + self.offy - np.floor(ly),self.frameShape[1]-self.stepXY),0)
        ry_high = min(max(np.floor(hy) + self.offy + np.ceil(ly),self.stepXY),self.frameShape[1])
        ry = np.arange(ry_low,ry_high).astype(int)
#        rx = (rx[np.minimum(rx>=0,rx<self.frameShape[1])]).astype(int)
#        ry = (ry[np.minimum(ry>=0,ry<self.frameShape[0])][:,np.newaxis]).astype(int)

        self.ax.imshow(im2show[rx,ry], vmin=vmin_tmp, vmax=vmax_tmp, interpolation='Nearest',extent=[ry[0],ry[-1]+1,rx[-1]+1,rx[0]])
        def format_coord(x, y):
            x = int(x)
            y = int(y)
            try:
                #return "%s @ [%4i, %4i]" % (round(im2show[y, x],2), x, y)
                return "%1e @ [%4i, %4i]" % (round(im2show[y, x],5), y, x) #first shown coordinate is vertical, second is horizontal
            except IndexError:
                return ""
        self.ax.format_coord = format_coord
        if 'qt' in pylab.matplotlib.get_backend().lower():
            self.fig.canvas.manager.window.raise_() #this pops the window to the top
        if self.showProfiles:
            posProf = self.posProfHoriz
            self.axX.cla()
            self.axX.plot(rx+1,im2show[posProf,rx])
#            plt.xlim(rx[0],rx[-1])
            self.axX.set_xlim(rx[0],rx[-1])
    def printStat(self):
        if self.globalScale:
            modePrint = 'all frames'
            img = self.im
            if self.complex:
                modePrint = 'modulus'
                img = self.im[...,0]

        else:
            if self.im.ndim > 2:
                img = self.im[...,self.i]
                modePrint = 'frame %g'%self.i
            else:
                img = self.im
                modePrint = 'frame'
        print ("\n-----")
        print ("Statistics of the " + modePrint + " in figure %g:"%self.figNum)
        print ("Shape: ", img.shape)
        print ("Maximum: ", img.max(), "@", np.unravel_index(np.argmax(img),img.shape))
        print ("Minimum: ", img.min(), "@", np.unravel_index(np.argmin(img),img.shape))
        print ("Center of mass:", nd.measurements.center_of_mass(img))
        print ("Mean: ", img.mean())
        print ("Standard deviation: ", img.std())
        print ("Variance: ", img.var())
        print ("Sum: ", img.sum())
        print ("Data type:", self.dtype)
        self.draw()
        self.fig.canvas.draw()


    def __call__(self, event):
#        old_i = self.i
        if event.key=='n':#'up': #'right'
            if self.im.ndim > 2:
                self.i = min(self.im.shape[2]-1, self.i+1)
        elif event.key == 'p':#'down': #'left'
            if self.im.ndim > 2:
                self.i = max(0, self.i-1)
        if event.key=='N':#'up': #'right'
            if self.im.ndim > 2:
                self.i = min(self.im.shape[2]-1, self.i+10)
        elif event.key == 'P':#'down': #'left'
            if self.im.ndim > 2:
                self.i = max(0, self.i-10)
        elif event.key == 'l':
            self.logon = np.mod(self.logon+1,2)
        elif event.key == 'c':
            self.cmap = np.mod(self.cmap+1,4)
        elif event.key in 'SMV':
            self.projToggle = np.mod(self.projToggle+1,2)
            self.projType = event.key
        elif event.key == 'i':
            if 4*self.zoom < min(self.im.shape[:1]): # 2*zoom must not be bigger than shape/2
                self.zoom = 2*self.zoom
        elif event.key == 'o':
            self.zoom = max(self.zoom/2,1)
        elif event.key == 'g':
            self.globalScale = np.mod(self.globalScale+1,2)
        elif event.key == 'right':
            self.offy += self.stepXY
            self.offy = min(self.offy,self.im.shape[0]-1)
        elif event.key == 'left':
            self.offy -= self.stepXY
            self.offy = max(self.offy,-self.im.shape[0]+1)

        elif event.key == 'down':
            self.offx += self.stepXY
            self.offx = min(self.offx,self.im.shape[1]-1)
        elif event.key == 'up':
            self.offx -= self.stepXY
            self.offx = max(self.offx,-self.im.shape[1]+1)
        elif event.key == 'r': # reset position to the center of the image
            self.offx,self.offy = 0,0
            print ("Reseting positions to the center.")
        elif event.key == 'R': # reset contrast
            self.offVmin,self.offVmax = 0,0
            print ("Reseting contrast.")
        elif event.key == 'q': # increase lower limit of the contrast
            self.offVmin = min(self.offVmin+.1,1)
        elif event.key == 'Q': # decrease lower limit of the contrast
            self.offVmin = max(self.offVmin-.1,0)
        elif event.key == 'w': # increase upper limit of the contrast
            self.offVmax = min(self.offVmax+.1,1)
        elif event.key == 'W': # decrease upper limit of the contrast
            self.offVmax = max(self.offVmax-.1,0)
#            print ("Increasing upper limit of the contrast: %g %% (press R to reset).\n"%round(self.offVmax*100))
        elif event.key == 'T': # print statistics of the whole dataset
            self.printStat(),
            print ("-----")
        elif event.key == 't': # print statistics of the current frame
            self.printStat(mode = 'current frame'),
            print ("-----")

#        if old_i != self.i:
#        print self.offx
        self.draw()
        self.fig.canvas.draw()

def im(my_img,ax=None,**kwargs):
    "Displays image showing the values under the cursor."
    if ax is None:
        ax = plt.gca()
    def format_coord(x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%s @ [%4i, %4i]" % (my_img[y, x], x, y)
        except IndexError:
            return ""
    ax.imshow(my_img,interpolation='nearest',**kwargs)
    ax.format_coord = format_coord
#    plt.colorbar()
    plt.draw()
    plt.show()

def imTiles(d,sizeX=None,titNum=True, cmap = 'gray', logdata=False,fs=20):
    "Displays the stack of images in the composed tiled figure."
    "To do: global scale"
    if sizeX==None:
        sizeX=np.ceil(np.sqrt(d.shape[0]))
    sizeY=np.ceil(d.shape[0]/sizeX)
    plt.figure(figsize=(sizeY, sizeX))
    for i in range(1,d.shape[0]+1):
        plt.subplot(sizeX,sizeY,i)
        image2show  = d[i-1]
        if logdata:
            image2show = np.log10(image2show)
        plt.imshow(image2show,interpolation='nearest',cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        if titNum:
            plt.title(str(i-1),fontsize=fs)
        plt.tight_layout()

def phase2rgb(s):
   """
   Crates RGB image with colour-coded phase.
   """
   ph = np.angle(s)
   t = np.pi/3
   nx,ny = s.shape
   rgba = np.zeros((nx,ny,4))
   rgba[:,:,0] = (ph<t)*(ph>-t) + (ph>t)*(ph<2*t)*(2*t-ph)/t + (ph>-2*t)*(ph<-t)*(ph+2*t)/t
   rgba[:,:,1] = (ph>t)         + (ph<-2*t)      *(-2*t-ph)/t+ (ph>0)*(ph<t)    *ph/t
   rgba[:,:,2] = (ph<-t)        + (ph>-t)*(ph<0) *(-ph)/t + (ph>2*t)         *(ph-2*t)/t
   return rgba

def complex2rgbalog(s,amin=0.5,dlogs=2):
   """
   Returns RGB image with colour-coded phases and log10(amplitude) in birghtness.
   from: pynx/ptycho
   """
   rgba = phase2rgb(s)
   a = np.log10(abs(s)+1e-20)
   a -= a.max()-dlogs # display dlogs orders of magnitude
   rgba[:,:,3] = amin+a/dlogs*(1-amin)*(a>0)
   return rgba

def complex2rgbalin(s):
   """
   Returns RGB image with with colour-coded phase and log10(amplitude) in birghtness.
   from: pynx/ptycho
   """
   rgba = phase2rgb(s)
   a = np.abs(s)
   a /= a.max()
   rgba[:,:,3] = a
   return rgba


def colorwheel(text_col='black', fs=16):
  """
  Color wheel for phases in hsv colormap.
  From: pyVincent/ptycho.py
  """
  xwheel=np.linspace(-1,1,100)
  ywheel=np.linspace(-1,1,100)[:,np.newaxis]
  rwheel=np.sqrt(xwheel**2+ywheel**2)
  phiwheel=-np.arctan2(ywheel,xwheel)  # Need the - sign because imshow starts at (top,left)
#  rhowheel=rwheel*np.exp(1j*phiwheel)
  rhowheel=1*np.exp(1j*phiwheel)
  plt.gca().set_axis_off()
  rgba=complex2rgbalin(rhowheel*(rwheel<1))
  plt.imshow(rgba,aspect='equal')
  plt.text(1.3, 0.5,'$0$', fontsize=fs, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color=text_col)
  plt.text(-.3, 0.5,'$\pi$', fontsize=fs, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color=text_col)

def insertColorwheel(left=.7, bottom=.2, width=.1, height=.1,text_col='black', fs=16):
    """
    Inserts color wheel to the current axis.
    """
    plt.axes((left,bottom,width,height), axisbg='w')
    colorwheel(text_col=text_col,fs=fs)


def insertColorbar(fig,im,left=.7, bottom=.1, width=.05, height=.8 )     :
    """
    Inserts color bar to the current axis.
    """
    cax = fig.add_axes((left,bottom,width,height), axisbg='w')
    plt.colorbar(im, cax=cax)


def showCplx(im,mask=0,pixSize_um=1,showGrid=True,amplitudeLog = False,maskPhase = False, maskPhaseThr = 0.01, cmapAmplitude = 'jet', cmapPhase = 'hsv', scalePhaseImg = True, suptit = None, fontSize=20, suptit_fontSize=10, show_what = 'phase_amplitude'):
    "Displays AMPLITUDE_PHASE or REAL_IMAG ('show_what') of the complex image in two subfigures."
    print (show_what.lower())
    if amplitudeLog:
        amplitude = np.log10(abs(im))
    else:
        amplitude = abs(im)
    phase = np.angle(im)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    #plt.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    #plt.imshow(abs(np.ma.masked_array(im,mask)))
    if show_what is 'real_imag':
        plt.imshow(im.real,extent=(0,im.shape[1]*pixSize_um,0,im.shape[0]*pixSize_um),cmap=cmapAmplitude,interpolation='Nearest')
    else:
        plt.imshow(amplitude,extent=(0,im.shape[1]*pixSize_um,0,im.shape[0]*pixSize_um),cmap=cmapAmplitude,interpolation='Nearest')
#    plt.colorbar(m)
    if showGrid:
        plt.grid(color='w')
    if pixSize_um !=1:
        plt.xlabel('microns',fontsize = fontSize)
        plt.ylabel('microns',fontsize = fontSize)
    if suptit == None:
        if show_what.lower() is 'real_imag':
            plt.title('Real',fontsize = fontSize)
        else:
            plt.title('Amplitude',fontsize = fontSize)
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)

#    position=f.add_axes([0.5,0.1,0.02,.8])  ## the parameters are the specified position you set
#    plt.colorbar(m,cax=position) ##
#    plt.setp(ax_cb.get_yticklabels(), visible=False)

    plt.subplot(122)
    if scalePhaseImg:
        vminPhase = -np.pi
        vmaxPhase = np.pi
    else:
        vminPhase = phase.min()
        vmaxPhase = phase.max()
    if show_what.lower() is 'real_imag':
        plt.imshow(im.imag,cmap=cmapPhase,interpolation='Nearest',extent=(0,im.shape[1]*pixSize_um,0,im.shape[0]*pixSize_um))
    else:
        plt.imshow(np.ma.masked_array(phase,mask),cmap=cmapPhase,interpolation='Nearest',vmin=vminPhase,vmax=vmaxPhase,extent=(0,im.shape[1]*pixSize_um,0,im.shape[0]*pixSize_um))
    if showGrid:
        plt.grid(color='k')
    if pixSize_um !=1:
        plt.xlabel('microns',fontsize = fontSize)
        plt.ylabel('microns',fontsize = fontSize)
    if suptit == None:
        if show_what.lower() is 'real_imag':
            plt.title('Imag',fontsize = fontSize)
        else:
            plt.title('Phase',fontsize = fontSize)
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    if cmapPhase == 'hsv':
        insertColorwheel(left=.85, fs=fontSize)
    if suptit != None:
        plt.suptitle(suptit,fontsize = suptit_fontSize)
    plt.tight_layout()

def showLog(im, cmap='jet'):
    "Displays log of the real image with correct colorbar."
    f = plt.figure();
    i = plt.imshow(im, norm=LogNorm(), cmap=cmap)
    f.colorbar(i)
    return f,i

def showFT(im,cmapAmplitude = 'jet', cmapPhase = 'hsv'):
    ftim = fftshift(fft2(fftshift(im)))

    plt.subplot(2,2,1)
    plt.imshow(abs(im),cmap=cmapAmplitude,interpolation='Nearest')
    plt.title('abs(Im)')

    plt.subplot(2,2,2)
    plt.imshow(np.angle(im),cmap=cmapPhase,interpolation='Nearest')
    plt.title('angle(Im)')

    plt.subplot(2,2,3)
    plt.imshow(np.log10(abs(ftim)),cmap=cmapAmplitude,interpolation='Nearest')
    plt.title('abs(fft2(im))')

    plt.subplot(2,2,4)
    plt.imshow(np.angle(ftim),cmap=cmapPhase,interpolation='Nearest')
    plt.title('angle(fft2(im))')

    if cmapPhase == 'hsv':
        insertColorwheel(left=.85)




def plotScan(val,x,y,verbose=True,xlab=None, ylab=None):
    plt.figure(); plt.scatter(x,y,c=val,s=40)
    if verbose:
        plt.plot(x,y)
        for i in range(len(x)):
            plt.annotate(str(i), xy = (x[i], y[i]), xytext = (5, 0), textcoords = 'offset points')
    plt.gca().invert_yaxis()
    plt.axes().set_aspect(1)
    plt.grid(True)
    plt.colorbar()
    plt.xlabel(xlab)
    plt.ylabel(ylab)

from colorsys import hls_to_rgb

def colorize(z):
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a,b in zip(A,B)]
    return c

def ca():
    """
    Close all windows.
    """
    plt.close('all')