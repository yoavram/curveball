import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import seaborn as sns

class Plate(object):
	#edge_color = '#888888'
	#bg_color = "#95a5a6"    

	def __init__(self, array):		
		self._array = array
		for i in range(self._array.shape[0]):
			for j in range(self._array.shape[1]):
				## convert to int if possible
				try:
					self._array[i,j] = int(self._array[i,j])
				except ValueError:
					pass			
		# self.width, self.height, self.nstrains = width, height, nstrains    
		
		# # Create the figure and axes
		# self.fig = plt.figure(figsize=((width + 2) / 3., (height + 2) / 3.))
		# self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9),
		# 							aspect='equal', frameon=False,
		# 							xlim=(-0.05, width + 0.05),
		# 							ylim=(-0.05, height + 0.05))
		# for axis in (self.ax.xaxis, self.ax.yaxis):
		# 	axis.set_major_formatter(plt.NullFormatter())
		# 	axis.set_major_locator(plt.NullLocator())

		# # Create the grid of squares
		# self.squares = np.array([[RegularPolygon((i + 0.5, j + 0.5),
		# 										 numVertices=4,
		# 										 radius=0.5 * np.sqrt(2),
		# 										 orientation=np.pi / 4,
		# 										 ec=self.edge_color,
		# 										 fc=self.bg_color)
		# 						  for j in range(height)]
		# 						 for i in range(width)])
		# [self.ax.add_patch(sq) for sq in self.squares.flat]  
		# self.strains = np.zeros((width, height), dtype=int)
		# # Create event hook for mouse clicks
		# self.fig.canvas.mpl_connect('button_press_event', self._button_press)
		

	#@classmethod
	#def ninety_six_wells(cls, nstrains=2):
#		return cls(12, 8, nstrains)


	@classmethod
	def from_csv(cls, filename):
		array = np.loadtxt(filename, dtype=object, delimiter=', ')
		return cls(array)
				
		
	def to_csv(self, fname):
		return np.savetxt(fname, self._array, fmt='%s', delimiter=', ')


	def to_array(self):
		return self._array.copy()
		#return np.rot90(self.strains)


	def __repr__(self):
		pass
		#return str(self.to_array())
  

		
	
	


if __name__ == '__main__':
	pass