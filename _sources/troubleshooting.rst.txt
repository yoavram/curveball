Troubleshooting
===============

1. I get an error: ``alloc: invalid block: 000000000932D1C0: 90 9``.

That's usually caused by a problem with the plotting library (matplotlib) GUI backend (Tk).
To switch to a different backend, create a file :file:`matplotlibrc` (it it doesn't already exist) in the folder
:file:`%HOME%\.matplotlib` on **Windows** or :file:`$HOME/.matplotlib` on Linux and OSX.
Then add this line to the file:

``backend      : qt4agg``

After adding this line, try running Curveball again. 
If you get an ``ImportError``, you probably need to install *pyqt* or *pyside* with whichever of the following that works:

>>> conda install pyqt 
>>> conda install pyside
>>> pip install pyqt 
>>> pip install pyside



