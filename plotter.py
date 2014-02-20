# from http://stackoverflow.com/questions/6620979/2d-slice-series-of-3d-array-in-numpy
import numpy as np
import pylab
pylab.rcParams['keymap.yscale'] = '' # to disable the binding of the key 'l'

class plotter:
    def __init__(self, im, i=0):
        self.im = np.dstack(im)
        self.i = i
        self.logon = 0
        self.cmap = 0
        self.vmin = im.min()
        self.vmax = im.max()
        self.fig = pylab.figure()
        #pylab.gray()
        self.ax = self.fig.add_subplot(111)
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
        if self.logon:
            tit += ' log'
            im2show = np.log10(im)
        else:
            im2show = im
        if self.cmap:
            pylab.gray()
        else:
            pylab.jet()
            
        self.ax.set_title(tit)

        pylab.show()

#        self.ax.imshow(im, vmin=self.vmin, vmax=self.vmax, interpolation=None)
        self.ax.imshow(im2show, interpolation='Nearest') 
        def format_coord(x, y):
            x = int(x + 0.5)
            y = int(y + 0.5)
            try:
                return "%s @ [%4i, %4i]" % (round(im2show[y, x],2), x, y)
            except IndexError:
                return ""
        self.ax.format_coord = format_coord


    def __call__(self, event):
#        old_i = self.i
        if event.key=='up': #'right'
            self.i = min(self.im.shape[2]-1, self.i+1)
        elif event.key == 'down': #'left'
            self.i = max(0, self.i-1)
        elif event.key == 'l':
            self.logon = np.mod(self.logon+1,2)            
        elif event.key == 'c':
            self.cmap = np.mod(self.cmap+1,2)
#        if old_i != self.i:            
        self.draw()
        self.fig.canvas.draw()


def slice_show(im, i=0):
    plotter(im, i)