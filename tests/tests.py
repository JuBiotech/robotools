import unittest
import numpy

from robotools import liquidhandling
from robotools import evotools
from robotools import janustools


class TestLabware(unittest.TestCase):
    def test_init(self):
        plate = liquidhandling.Labware('TestPlate', 2, 3, 50, 250, initial_volumes=30)
        self.assertEqual(plate.name, 'TestPlate')
        self.assertTrue(numpy.array_equal(plate.volumes, numpy.array([
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
            assert plate.volumes[plate.indices[well]] == 150
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
        self.assertTrue(numpy.array_equal(plate.volumes, numpy.array([
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
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [150,200,200,200],
                [150,200,200,200],
                [200,200,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [50,0,0,0],
                [50,0,0,0],
                [0,0,0,0],
            ])))
            worklist.transfer(A, ['A03', 'B04'], B, ['A04', 'B04'], 50)
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [150,200,150,200],
                [150,200,200,150],
                [200,200,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
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
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [125,200,200,200],
                [200,200,200,200],
                [200,200,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
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
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [175,175,175,200],
                [200,200,200,200],
                [200,200,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [0,0,0,0],
                [75,0,0,0],
                [0,0,0,0],
            ])))
        return

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label=None, position=1, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label=15, position=1, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='thisisaveryverylongracklabelthatexceedsthemaximumlength', position=1, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='rack label; with semicolon', position=1, volume=15)
        evotools._prepate_aspirate_dispense_parameters(rack_label='valid rack label', position=1, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=None, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position='3', volume=15)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=-1, volume=15)
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=None)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume='nan')
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=float('nan'))
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=-15.4)
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume='15')
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=20)
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=23.78)
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=numpy.array(23.4))
        
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, liquid_class=None)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, liquid_class='thisisaveryverylongliquidclassthatexceedsthemaximumlength')
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, liquid_class='liquid;class')
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, liquid_class='valid liquid class')
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, tip=None)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, tip=12)
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, tip=4)
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, tip=evotools.Tip.T5)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, rack_id=None)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, rack_id='invalid;rack')
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, rack_id='thisisaveryverylongrackthatexceedsthemaximumlength')
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, rack_id='1235464')
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, rack_type=None)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, rack_type='invalid;rack type')
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, rack_type='thisisaveryverylongracktypethatexceedsthemaximumlength')
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, rack_type='valid rack type')
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, forced_rack_type=None)
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, forced_rack_type='invalid;forced rack type')
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, forced_rack_type='thisisaveryverylongforcedracktypethatexceedsthemaximumlength')
        evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume=15, forced_rack_type='valid forced rack type')
        return

    def test_comment(self):
        with evotools.Worklist() as wl:
            wl.comment('This is a simple comment')
            with self.assertRaises(ValueError):
                wl.comment('It must not contain ; semicolons')
            wl.comment("""
            But it may very well be
            a multiline comment
            """)
            self.assertEqual(wl, [
                'C;This is a simple comment',
                'C;But it may very well be',
                'C;a multiline comment'
            ])
        return

    def test_wash(self):
        with evotools.Worklist() as wl:
            wl.wash()
            with self.assertRaises(ValueError):
                wl.wash(scheme=15)
            with self.assertRaises(ValueError):
                wl.wash(scheme='2')
            wl.wash(scheme=1)
            wl.wash(scheme=2)
            wl.wash(scheme=3)
            wl.wash(scheme=4)
            self.assertEqual(wl, [
                'W1;',
                'W1;',
                'W2;',
                'W3;',
                'W4;',
            ])
        return

    def test_decontaminate(self):
        with evotools.Worklist() as wl:
            wl.decontaminate()
            self.assertEqual(wl, [
                'WD;',
            ])
        return

    def test_flush(self):
        with evotools.Worklist() as wl:
            wl.flush()
            self.assertEqual(wl, [
                'F;',
            ])
        return

    def test_commit(self):
        with evotools.Worklist() as wl:
            wl.commit()
            self.assertEqual(wl, [
                'B;',
            ])
        return

    def test_set_diti(self):
        with evotools.Worklist() as wl:
            wl.set_diti(diti_index=1)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.set_diti(diti_index=2)
            wl.commit()
            wl.set_diti(diti_index=2)
            self.assertEqual(wl, [
                'S;1',
                'B;',
                'S;2',
            ])
        return

    def test_aspirate_raw(self):
        with evotools.Worklist() as wl:
            wl._aspirate('WaterTrough', 1, 200)
            self.assertEqual(wl[-1], 'A;WaterTrough;;;1;;200.0;;;;')
            wl._aspirate('WaterTrough', 1, 200, rack_id='12345', rack_type='my_rack_id', tube_id='my_tube_id')
            self.assertEqual(wl[-1], 'A;WaterTrough;12345;my_rack_id;1;my_tube_id;200.0;;;;')
            wl._aspirate('WaterTrough', 1, 200, liquid_class='my_liquid_class', tip=8, forced_rack_type='forced_rack')
            self.assertEqual(wl[-1], 'A;WaterTrough;;;1;;200.0;my_liquid_class;;128;forced_rack')
        return

    def test_dispense_raw(self):
        with evotools.Worklist() as wl:
            wl._dispense('WaterTrough', 1, 200)
            self.assertEqual(wl[-1], 'D;WaterTrough;;;1;;200.0;;;;')
            wl._dispense('WaterTrough', 1, 200, rack_id='12345', rack_type='my_rack_id', tube_id='my_tube_id')
            self.assertEqual(wl[-1], 'D;WaterTrough;12345;my_rack_id;1;my_tube_id;200.0;;;;')
            wl._dispense('WaterTrough', 1, 200, liquid_class='my_liquid_class', tip=8, forced_rack_type='forced_rack')
            self.assertEqual(wl[-1], 'D;WaterTrough;;;1;;200.0;my_liquid_class;;128;forced_rack')
        return

    def test_aspirate(self):
        source = liquidhandling.Labware('SourceLW', rows=3, columns=2, min_volume=10, max_volume=200, initial_volumes=200)
        destination = liquidhandling.Labware('DestinationLW', rows=2, columns=3, min_volume=10, max_volume=200)
        with evotools.Worklist() as wl:
            wl.aspirate(source, ['A01', 'A02', 'C02'], 150)
            wl.dispense(destination, ['A01', 'A02', 'A03'], 150)
            self.assertEqual(wl, [
                'A;SourceLW;;;1;;150.0;;;;',
                'A;SourceLW;;;4;;150.0;;;;',
                'A;SourceLW;;;6;;150.0;;;;',
                'D;DestinationLW;;;1;;150.0;;;;',
                'D;DestinationLW;;;3;;150.0;;;;',
                'D;DestinationLW;;;5;;150.0;;;;',
            ])
            pass
        return