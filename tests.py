import unittest
import numpy

from robotools import liquidhandling
from robotools import evotools
from robotools import janustools


class TestLabware(unittest.TestCase):
    def test_init(self):
        plate = liquidhandling.Labware('TestPlate', 2, 3, 50, 250, initial_volumes=30)
        self.assertEqual(plate.name, 'TestPlate')
        self.assertTrue(numpy.array_equal(plate.current, numpy.array([
            [30,30,30],
            [30,30,30]
        ])))
        self.assertEqual(plate.min_volume, 50)
        self.assertEqual(plate.max_volume, 250)
        

class TestLabwareAddRemove(unittest.TestCase):
    def test_add_valid(self):
        plate = liquidhandling.Labware('TestPlate', 4, 6, 100, 250)
        wells = ['A01', 'A02', 'B04']
        plate.add(wells, 150)
        for well in wells:
            assert plate.current[plate.indices[well]] == 150
        return
    
    def test_add_too_much(self):
        plate = liquidhandling.Labware('TestPlate', 4, 6, 100, 250)
        wells = ['A01', 'A02', 'B04']
        with self.assertRaises(liquidhandling.VolumeOverflowError):
            plate.add(wells, 500)
        return

    def test_remove_valid(self):
        plate = liquidhandling.Labware('TestPlate', 2, 3, 50, 250, initial_volumes=200)
        wells = ['A01', 'A02', 'B03']
        plate.remove(wells, 50)
        self.assertTrue(numpy.array_equal(plate.current, numpy.array([
            [150,150,200],
            [200,200,150]
        ])))
        return
    
    def test_remove_too_much(self):
        plate = liquidhandling.Labware('TestPlate', 4, 6, 100, 250)
        wells = ['A01', 'A02', 'B04']
        with self.assertRaises(liquidhandling.VolumeUnderflowError):
            plate.remove(wells, 500)
        return
    
    
class TestWorklist(unittest.TestCase):
    def test_context(self):
        with evotools.Worklist() as worklist:
            self.assertIsNotNone(worklist)
        return
    
    def test_transfer_many_many(self):
        A = liquidhandling.Labware('A', 3, 4, 50, 250, initial_volumes=200)
        B = liquidhandling.Labware('A', 3, 4, 50, 250)
        wells = ['A01', 'B01']
        with evotools.Worklist() as worklist:
            worklist.transfer(A, wells, B, wells, 50)
            self.assertTrue(numpy.array_equal(A.current, numpy.array([
                [150,200,200,200],
                [150,200,200,200],
                [200,200,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.current, numpy.array([
                [50,0,0,0],
                [50,0,0,0],
                [0,0,0,0],
            ])))
            worklist.transfer(A, ['A03', 'B04'], B, ['A04', 'B04'], 50)
            self.assertTrue(numpy.array_equal(A.current, numpy.array([
                [150,200,150,200],
                [150,200,200,150],
                [200,200,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.current, numpy.array([
                [50,0,0,50],
                [50,0,0,50],
                [0,0,0,0],
            ])))
        return
    
    def test_transfer_one_many(self):
        A = liquidhandling.Labware('A', 3, 4, 50, 250, initial_volumes=200)
        B = liquidhandling.Labware('A', 3, 4, 50, 250)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, ['A01'], B, ['B01', 'B02', 'B03'], 25)
            self.assertTrue(numpy.array_equal(A.current, numpy.array([
                [125,200,200,200],
                [200,200,200,200],
                [200,200,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.current, numpy.array([
                [0,0,0,0],
                [25,25,25,0],
                [0,0,0,0],
            ])))
        return
    
    def test_transfer_many_one(self):
        A = liquidhandling.Labware('A', 3, 4, 50, 250, initial_volumes=200)
        B = liquidhandling.Labware('A', 3, 4, 50, 250)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, ['A01', 'A02', 'A03'], B, 'B01', 25)
            self.assertTrue(numpy.array_equal(A.current, numpy.array([
                [175,175,175,200],
                [200,200,200,200],
                [200,200,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.current, numpy.array([
                [0,0,0,0],
                [75,0,0,0],
                [0,0,0,0],
            ])))
        return

