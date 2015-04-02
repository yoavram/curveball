import unittest
from curveball import Plate
import numpy as np
import tempfile
import os
import filecmp


fname = 'tests/plate.csv'


class PlateTest(unittest.TestCase):
	def setUp(self):
		self.array = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0]])
		np.savetxt(fname, self.array, fmt='%s', delimiter=', ')


	def tearDown(self):
		os.remove(fname)


	def test_init(self):
		plate = Plate(self.array)
		self.assertTrue((plate._array == self.array).all())


	def test_from_csv(self):
		plate  = Plate.from_csv(fname)
		self.assertTrue((plate._array == self.array).all())


	def test_to_csv(self):
		plate = Plate(self.array)
		output = tempfile.NamedTemporaryFile(delete=False)
		plate.to_csv(output.name)
		self.assertTrue(filecmp.cmp(output.name, fname))
		output.close()	
		os.remove(output.name)


	def test_to_array(self):
		plate  = Plate(self.array)
		arr = plate.to_array()
		self.assertTrue((arr == self.array).all())
		self.assertFalse(arr is self.array)


	def test_get_strains(self):
		plate  = Plate(self.array)
		strains = plate.strains
		self.assertEquals(strains,[0,1,2,3,4])


	def test_colormap(self):
		plate  = Plate(self.array)
		colormap = plate.colormap
		self.assertEquals(sorted(colormap.keys()), plate.strains)
		self.assertEquals(colormap[0], Plate.BLANK_COLOR)
		colors = sorted(colormap.values())
		colorset = sorted(set(colors))
		self.assertEquals(colors, colorset)


if __name__ == '__main__':
    unittest.main()