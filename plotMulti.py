# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:41:28 2013

@author: ondrej
"""
import Tkinter
import Image, ImageTk
import numpy

def button_click_exit_mainloop (event):
    event.widget.quit() # this will cause mainloop to unblock.

root = Tkinter.Tk()
root.bind("<Button>", button_click_exit_mainloop)
root.geometry('+%d+%d' % (100,100))
d=numpy.random.rand(6,2,3)
old_label_image = None
for f in range(1,d.shape[1]):
    try:
        print f
        image1 = Image.open(f)
        root.geometry('%dx%d' % (image1.size[0],image1.size[1]))
        tkpi = ImageTk.PhotoImage(image1)
        label_image = Tkinter.Label(root, image=tkpi)
        label_image.place(x=0,y=0,width=image1.size[0],height=image1.size[1])
        root.title(f)
        if old_label_image is not None:
            old_label_image.destroy()
        old_label_image = label_image
        root.mainloop() # wait until user clicks the window
    except Exception, e:
        # This is used to skip anything not an image.
        # Image.open will generate an exception if it cannot open a file.
        # Warning, this will hide other errors as well.
        pass