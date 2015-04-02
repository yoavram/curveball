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
		os.remove(output.name)


	def test_to_array(self):
		plate  = Plate(self.array)
		arr = plate.to_array()
		self.assertTrue((arr == self.array).all())
		self.assertFalse(arr is self.array)


if __name__ == '__main__':
    unittest.main()