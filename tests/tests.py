import unittest
import numpy
import os
import tempfile

from robotools import liquidhandling
from robotools import evotools
from robotools import janustools


class TestStandardLabware(unittest.TestCase):
    def test_init(self):
        plate = liquidhandling.Labware('TestPlate', 2, 3, min_volume=50, max_volume=250, initial_volumes=30)
        self.assertEqual(plate.name, 'TestPlate')
        self.assertEqual(plate.row_ids, tuple('AB'))
        self.assertEqual(plate.column_ids, [1,2,3])
        self.assertEqual(plate.n_rows, 2)
        self.assertEqual(plate.n_columns, 3)
        self.assertEqual(plate.min_volume, 50)
        self.assertEqual(plate.max_volume, 250)
        self.assertEqual(len(plate.history), 1)
        self.assertTrue(numpy.array_equal(plate.volumes, numpy.array([
            [30,30,30],
            [30,30,30]
        ])))
        self.assertDictEqual(plate.indices, {
            'A01': (0, 0), 'A02': (0, 1), 'A03': (0, 2),
            'B01': (1, 0), 'B02': (1, 1), 'B03': (1, 2),
        })
        self.assertDictEqual(plate.positions, {
            'A01': 1, 'A02': 3, 'A03': 5,
            'B01': 2, 'B02': 4, 'B03': 6,
        })
        return

    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            _ = liquidhandling.Labware('A', 0, 3, min_volume=10, max_volume=250)
        with self.assertRaises(ValueError):
            _ = liquidhandling.Labware('A', 3, 0, min_volume=10, max_volume=250)
        with self.assertRaises(ValueError):
            _ = liquidhandling.Labware('A', 3, 4, min_volume=10, max_volume=250, virtual_rows=2)
        with self.assertRaises(ValueError):
            _ = liquidhandling.Labware('A', 1, 4, min_volume=10, max_volume=250, virtual_rows=0)
        return

    def test_volume_limits(self):
        with self.assertRaises(ValueError):
            _ = liquidhandling.Labware('A', 3, 4, min_volume=-30, max_volume=100)
        with self.assertRaises(ValueError):
            _ = liquidhandling.Labware('A', 3, 4, min_volume=100, max_volume=70)
        with self.assertRaises(ValueError):
            _ = liquidhandling.Labware('A', 3, 4, min_volume=10, max_volume=70, initial_volumes=100)
        with self.assertRaises(ValueError):
            _ = liquidhandling.Labware('A', 3, 4, min_volume=10, max_volume=70, initial_volumes=-10)
        _ = liquidhandling.Labware('A', 3, 4, min_volume=10, max_volume=70, initial_volumes=50)
        return

    def test_initial_volumes(self):
        plate = liquidhandling.Labware('TestPlate', 1, 3, min_volume=50, max_volume=250, initial_volumes=[20,30,40])
        self.assertTrue(numpy.array_equal(plate.volumes, numpy.array([
            [20,30,40],
        ])))
        return

    def test_logging(self):
        plate = liquidhandling.Labware('TestPlate', 2, 3, min_volume=50, max_volume=250)
        plate.add(plate.wells, 25)
        plate.add(plate.wells, 25)
        plate.add(plate.wells, 25)
        plate.add(plate.wells, 25)
        self.assertEqual(len(plate.history), 5)
        return

    def test_log_condensation_first(self):
        plate = liquidhandling.Labware('TestPlate', 2, 3, min_volume=50, max_volume=250)
        plate.add(plate.wells, 25, label='A')
        plate.add(plate.wells, 25, label='B')
        plate.add(plate.wells, 25, label='C')
        plate.add(plate.wells, 25, label='D')
        self.assertEqual(len(plate.history), 5)

        # condense the last two as 'D'
        plate.condense_log(2, label='last')
        self.assertEqual(len(plate.history), 4)
        self.assertEqual(plate.history[-1][0], 'D')
        self.assertTrue(numpy.array_equal(plate.history[-1][1], numpy.array([
            [100,100,100],
            [100,100,100],
        ])))

        # condense the last three as 'A'
        plate.condense_log(3, label='first')
        self.assertEqual(len(plate.history), 2)
        self.assertEqual(plate.history[-1][0], 'A')
        self.assertTrue(numpy.array_equal(plate.history[-1][1], numpy.array([
            [100,100,100],
            [100,100,100],
        ])))

        # condense the remaining two as 'prepared'
        plate.condense_log(3, label='prepared')
        self.assertEqual(len(plate.history), 1)
        self.assertEqual(plate.history[-1][0], 'prepared')
        self.assertTrue(numpy.array_equal(plate.history[-1][1], numpy.array([
            [100,100,100],
            [100,100,100],
        ])))
        return

    def test_add_valid(self):
        plate = liquidhandling.Labware('TestPlate', 4, 6, min_volume=100, max_volume=250)
        wells = ['A01', 'A02', 'B04']
        plate.add(wells, 150)
        plate.add(wells, 3.5)
        self.assertEqual(len(plate.history), 3)
        for well in wells:
            assert plate.volumes[plate.indices[well]] == 153.5
        return
    
    def test_add_too_much(self):
        plate = liquidhandling.Labware('TestPlate', 4, 6, min_volume=100, max_volume=250)
        wells = ['A01', 'A02', 'B04']
        with self.assertRaises(liquidhandling.VolumeOverflowError):
            plate.add(wells, 500)
        return

    def test_remove_valid(self):
        plate = liquidhandling.Labware('TestPlate', 2, 3, min_volume=50, max_volume=250, initial_volumes=200)
        wells = ['A01', 'A02', 'B03']
        plate.remove(wells, 50)
        self.assertEqual(len(plate.history), 2)
        self.assertTrue(numpy.array_equal(plate.volumes, numpy.array([
            [150,150,200],
            [200,200,150]
        ])))
        return
    
    def test_remove_too_much(self):
        plate = liquidhandling.Labware('TestPlate', 4, 6, min_volume=100, max_volume=250)
        wells = ['A01', 'A02', 'B04']
        with self.assertRaises(liquidhandling.VolumeUnderflowError):
            plate.remove(wells, 500)
        self.assertEqual(len(plate.history), 1)
        return
        

class TestTroughLabware(unittest.TestCase):
    def test_init_trough(self):
        trough = liquidhandling.Labware('TestTrough', 1, 4, min_volume=1000, max_volume=50*1000, initial_volumes=30*1000, virtual_rows=5)
        self.assertEqual(trough.name, 'TestTrough')
        self.assertEqual(trough.row_ids, tuple('ABCDE'))
        self.assertEqual(trough.column_ids, [1,2,3,4])
        self.assertEqual(trough.min_volume, 1000)
        self.assertEqual(trough.max_volume, 50*1000)
        self.assertEqual(len(trough.history), 1)
        self.assertTrue(numpy.array_equal(trough.volumes, numpy.array([
            [30*1000,30*1000,30*1000,30*1000]
        ])))
        self.assertDictEqual(trough.indices, {
            'A01': (0, 0), 'A02': (0, 1), 'A03': (0, 2), 'A04': (0, 3),
            'B01': (0, 0), 'B02': (0, 1), 'B03': (0, 2), 'B04': (0, 3),
            'C01': (0, 0), 'C02': (0, 1), 'C03': (0, 2), 'C04': (0, 3),
            'D01': (0, 0), 'D02': (0, 1), 'D03': (0, 2), 'D04': (0, 3),
            'E01': (0, 0), 'E02': (0, 1), 'E03': (0, 2), 'E04': (0, 3),
        })
        self.assertDictEqual(trough.positions, {
            'A01': 1, 'A02': 6, 'A03': 11, 'A04': 16,
            'B01': 2, 'B02': 7, 'B03': 12, 'B04': 17,
            'C01': 3, 'C02': 8, 'C03': 13, 'C04': 18,
            'D01': 4, 'D02': 9, 'D03': 14, 'D04': 19,
            'E01': 5, 'E02': 10, 'E03': 15, 'E04': 20,
        })
        return

    def test_initial_volumes(self):
        trough = liquidhandling.Labware('TestTrough', 1, 4, min_volume=1000, max_volume=50*1000, initial_volumes=[30*1000, 20*1000, 20*1000, 20*1000], virtual_rows=5)
        self.assertTrue(numpy.array_equal(trough.volumes, numpy.array([
            [30*1000, 20*1000, 20*1000, 20*1000],
        ])))
        return

    def test_trough_add_valid(self):
        trough = liquidhandling.Labware('TestTrough', 1, 4, min_volume=100, max_volume=250, virtual_rows=3)
        # adding into the first column (which is actually one well)
        trough.add(['A01', 'B01'], 50)
        self.assertTrue(numpy.array_equal(trough.volumes, numpy.array([
            [100, 0, 0, 0]
        ])))
        # adding to the last row (separate wells)
        trough.add(['C01', 'C02', 'C03'], 50)
        self.assertTrue(numpy.array_equal(trough.volumes, numpy.array([
            [150, 50, 50, 0]
        ])))
        self.assertEqual(len(trough.history), 3)
        return

    def test_trough_add_too_much(self):
        trough = liquidhandling.Labware('TestTrough', 1, 4, min_volume=100, max_volume=1000, virtual_rows=3)
        # adding into the first column (which is actually one well)
        with self.assertRaises(liquidhandling.VolumeOverflowError):
            trough.add(['A01', 'B01'], 600)
        return

    def test_trough_remove_valid(self):
        trough = liquidhandling.Labware('TestTrough', 1, 4, min_volume=1000, max_volume=30000, virtual_rows=3, initial_volumes=3000)
        # adding into the first column (which is actually one well)
        trough.remove(['A01', 'B01'], 50)
        self.assertTrue(numpy.array_equal(trough.volumes, numpy.array([
            [2900, 3000, 3000, 3000]
        ])))
        # adding to the last row (separate wells)
        trough.remove(['C01', 'C02', 'C03'], 50)
        self.assertTrue(numpy.array_equal(trough.volumes, numpy.array([
            [2850, 2950, 2950, 3000]
        ])))
        self.assertEqual(len(trough.history), 3)
        return

    def test_trough_remove_too_much(self):
        trough = liquidhandling.Labware('TestTrough', 1, 4, min_volume=1000, max_volume=30*1000, virtual_rows=3, initial_volumes=3000)
        # adding into the first column (which is actually one well)
        with self.assertRaises(liquidhandling.VolumeUnderflowError):
            trough.remove(['A01', 'B01'], 2000)
        return
        
    
class TestWorklist(unittest.TestCase):
    def test_context(self):
        with evotools.Worklist() as worklist:
            self.assertIsNotNone(worklist)
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
        with self.assertRaises(ValueError):
            evotools._prepate_aspirate_dispense_parameters(rack_label='WaterTrough', position=1, volume='bla')
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
            # empty and None comments should be ignored
            wl.comment('')
            wl.comment(None)
            # this will be the first actual comment
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

    def test_aspirate_single(self):
        with evotools.Worklist() as wl:
            wl.aspirate_well('WaterTrough', 1, 200)
            self.assertEqual(wl[-1], 'A;WaterTrough;;;1;;200.00;;;;')
            wl.aspirate_well('WaterTrough', 1, 200, rack_id='12345', rack_type='my_rack_id', tube_id='my_tube_id')
            self.assertEqual(wl[-1], 'A;WaterTrough;12345;my_rack_id;1;my_tube_id;200.00;;;;')
            wl.aspirate_well('WaterTrough', 1, 200, liquid_class='my_liquid_class', tip=8, forced_rack_type='forced_rack')
            self.assertEqual(wl[-1], 'A;WaterTrough;;;1;;200.00;my_liquid_class;;128;forced_rack')
        return

    def test_dispense_single(self):
        with evotools.Worklist() as wl:
            wl.dispense_well('WaterTrough', 1, 200)
            self.assertEqual(wl[-1], 'D;WaterTrough;;;1;;200.00;;;;')
            wl.dispense_well('WaterTrough', 1, 200, rack_id='12345', rack_type='my_rack_id', tube_id='my_tube_id')
            self.assertEqual(wl[-1], 'D;WaterTrough;12345;my_rack_id;1;my_tube_id;200.00;;;;')
            wl.dispense_well('WaterTrough', 1, 200, liquid_class='my_liquid_class', tip=8, forced_rack_type='forced_rack')
            self.assertEqual(wl[-1], 'D;WaterTrough;;;1;;200.00;my_liquid_class;;128;forced_rack')
        return

    def test_aspirate_systemliquid(self):
        with evotools.Worklist() as wl:
            wl.aspirate_well(evotools.Labwares.SystemLiquid, 1, 200)
            self.assertEqual(wl[-1], 'A;Systemliquid;;;1;;200.00;;;;')
        return

    def test_save(self):
        tf = tempfile.mktemp() + '.gwl'
        error = None
        try:
            with evotools.Worklist() as worklist:
                worklist.flush()
                worklist.save(tf)
                self.assertTrue(os.path.exists(tf))
                # also check that the file can be overwritten if it exists already
                worklist.save(tf)
            self.assertTrue(os.path.exists(tf))
            with open(tf) as file:
                lines = file.readlines()
                self.assertEqual(lines, [
                    'F;'
                ])
        except Exception as ex:
            error = ex
        finally:
            os.remove(tf)
        self.assertFalse(os.path.exists(tf))
        if error:
            raise error
        return

    def test_autosave(self):
        tf = tempfile.mktemp() + '.gwl'
        error = None
        try:
            with evotools.Worklist(tf) as worklist:
                worklist.flush()
            self.assertTrue(os.path.exists(tf))
            with open(tf) as file:
                lines = file.readlines()
                self.assertEqual(lines, [
                    'F;'
                ])
        except Exception as ex:
            error = ex
        finally:
            os.remove(tf)
        self.assertFalse(os.path.exists(tf))
        if error:
            raise error
        return

    def test_reagent_distribution(self):
        with evotools.Worklist() as wl:
            with self.assertRaises(NotImplementedError):
                wl._reagent_distribution()
        return

class TestStandardLabwareWorklist(unittest.TestCase):
    def test_aspirate(self):
        source = liquidhandling.Labware('SourceLW', rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200)
        with evotools.Worklist() as wl:
            wl.aspirate(source, ['A01', 'A02', 'C02'], 50, label=None)
            wl.aspirate(source, ['A03', 'B03', 'C03'], [10,20,30.5], label='second aspirate')
            self.assertEqual(wl, [
                'A;SourceLW;;;1;;50.00;;;;',
                'A;SourceLW;;;4;;50.00;;;;',
                'A;SourceLW;;;6;;50.00;;;;',
                'C;second aspirate',
                'A;SourceLW;;;7;;10.00;;;;',
                'A;SourceLW;;;8;;20.00;;;;',
                'A;SourceLW;;;9;;30.50;;;;',
            ])
            self.assertTrue(numpy.array_equal(source.volumes, [
                [150,150,190],
                [200,200,180],
                [200,150,169.5],
            ]))
            self.assertEqual(len(source.history), 3)
        return

    def test_aspirate_2d_volumes(self):
        source = liquidhandling.Labware('SourceLW', rows=2, columns=3, min_volume=10, max_volume=200, initial_volumes=200)
        with evotools.Worklist() as wl:
            wl.aspirate(source, source.wells[:,:2], volumes=numpy.array([
                [20,30],
                [15.3,17.53],
            ]))
            self.assertEqual(wl, [
                'A;SourceLW;;;1;;20.00;;;;',
                'A;SourceLW;;;2;;15.30;;;;',
                'A;SourceLW;;;3;;30.00;;;;',
                'A;SourceLW;;;4;;17.53;;;;',
            ])
            self.assertTrue(numpy.array_equal(source.volumes, [
                [180,170,200],
                [200-15.3,200-17.53,200]
            ]))
            self.assertEqual(len(source.history), 2)
        return

    def test_dispense(self):
        destination = liquidhandling.Labware('DestinationLW', rows=2, columns=3, min_volume=10, max_volume=200)
        with evotools.Worklist() as wl:
            wl.dispense(destination, ['A01', 'A02', 'A03'], 150, label=None)
            wl.dispense(destination, ['B01', 'B02', 'B03'], [10,20,30.5], label='second dispense')
            self.assertEqual(wl, [
                'D;DestinationLW;;;1;;150.00;;;;',
                'D;DestinationLW;;;3;;150.00;;;;',
                'D;DestinationLW;;;5;;150.00;;;;',
                'C;second dispense',
                'D;DestinationLW;;;2;;10.00;;;;',
                'D;DestinationLW;;;4;;20.00;;;;',
                'D;DestinationLW;;;6;;30.50;;;;',
            ])            
            self.assertTrue(numpy.array_equal(destination.volumes, [
                [150,150,150],
                [10,20,30.5],
            ]))
            self.assertEqual(len(destination.history), 3)
        return

    def test_dispense_2d_volumes(self):
        destination = liquidhandling.Labware('DestinationLW', rows=2, columns=3, min_volume=10, max_volume=200)
        with evotools.Worklist() as wl:
            wl.dispense(destination, destination.wells[:,:2], volumes=numpy.array([
                [20,30],
                [15.3,17.53],
            ]))
            self.assertEqual(wl, [
                'D;DestinationLW;;;1;;20.00;;;;',
                'D;DestinationLW;;;2;;15.30;;;;',
                'D;DestinationLW;;;3;;30.00;;;;',
                'D;DestinationLW;;;4;;17.53;;;;',
            ])            
            self.assertTrue(numpy.array_equal(destination.volumes, [
                [20,30,0],
                [15.3,17.53,0]
            ]))
            self.assertEqual(len(destination.history), 2)
        return

    def test_skip_zero_volumes(self):
        source = liquidhandling.Labware('SourceLW', rows=3, columns=3, min_volume=10, max_volume=200, initial_volumes=200)
        destination = liquidhandling.Labware('DestinationLW', rows=2, columns=3, min_volume=10, max_volume=200)
        with evotools.Worklist() as wl:
            wl.aspirate(source, ['A03', 'B03', 'C03'], [10,0,30.5])
            wl.dispense(destination, ['B01', 'B02', 'B03'], [10,0,30.5])
            self.assertEqual(wl, [
                'A;SourceLW;;;7;;10.00;;;;',
                'A;SourceLW;;;9;;30.50;;;;',
                'D;DestinationLW;;;2;;10.00;;;;',
                'D;DestinationLW;;;6;;30.50;;;;',
            ])
            self.assertTrue(numpy.array_equal(source.volumes, [
                [200,200,190],
                [200,200,200],
                [200,200,169.5],
            ]))
            self.assertTrue(numpy.array_equal(destination.volumes, [
                [0,0,0],
                [10,0,30.5],
            ]))
            self.assertEqual(len(destination.history), 2)
        return

    def test_transfer_2d_volumes(self):
        A = liquidhandling.Labware('A', 2, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware('B', 2, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as wl:
            wl.transfer(
                A, A.wells[:,:2],
                B, B.wells[:,:2], volumes=numpy.array([
                    [20,30],
                    [15.3,17.53],
                ]),
            )
            self.assertEqual(wl, [
                'A;A;;;1;;20.00;;;;',
                'D;B;;;1;;20.00;;;;',
                'W1;',
                'A;A;;;2;;15.30;;;;',
                'D;B;;;2;;15.30;;;;',
                'W1;',
                'A;A;;;3;;30.00;;;;',
                'D;B;;;3;;30.00;;;;',
                'W1;',
                'A;A;;;4;;17.53;;;;',
                'D;B;;;4;;17.53;;;;',
                'W1;',
            ])            
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [180,170,200,200],
                [200-15.3,200-17.53,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [20,30,0,0],
                [15.3,17.53,0,0],
            ])))
            self.assertEqual(len(A.history), 2)
            self.assertEqual(len(B.history), 2)
        return

    def test_transfer_2d_volumes_no_wash(self):
        A = liquidhandling.Labware('A', 2, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware('B', 2, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as wl:
            wl.transfer(
                A, A.wells[:,:2],
                B, B.wells[:,:2], volumes=numpy.array([
                    [20,30],
                    [15.3,17.53],
                ]),
                wash_scheme=None
            )
            self.assertEqual(wl, [
                'A;A;;;1;;20.00;;;;',
                'D;B;;;1;;20.00;;;;',
                'A;A;;;2;;15.30;;;;',
                'D;B;;;2;;15.30;;;;',
                'A;A;;;3;;30.00;;;;',
                'D;B;;;3;;30.00;;;;',
                'A;A;;;4;;17.53;;;;',
                'D;B;;;4;;17.53;;;;',
            ])            
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [180,170,200,200],
                [200-15.3,200-17.53,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [20,30,0,0],
                [15.3,17.53,0,0],
            ])))
            self.assertEqual(len(A.history), 2)
            self.assertEqual(len(B.history), 2)
        return

    def test_transfer_many_many(self):
        A = liquidhandling.Labware('A', 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware('B', 3, 4, min_volume=50, max_volume=250)
        wells = ['A01', 'B01']
        with evotools.Worklist() as worklist:
            worklist.transfer(A, wells, B, wells, 50, label='first transfer')
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
            worklist.transfer(A, ['A03', 'B04'], B, ['A04', 'B04'], 50, label='second transfer')
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
            self.assertEqual(worklist, [
                'C;first transfer',
                'A;A;;;1;;50.00;;;;',
                'D;B;;;1;;50.00;;;;',
                'W1;',
                'A;A;;;2;;50.00;;;;',
                'D;B;;;2;;50.00;;;;',
                'W1;',
                'C;second transfer',
                'A;A;;;7;;50.00;;;;',
                'D;B;;;10;;50.00;;;;',
                'W1;',
                'A;A;;;11;;50.00;;;;',
                'D;B;;;11;;50.00;;;;',
                'W1;',
            ])
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return

    def test_transfer_many_many_2d(self):
        A = liquidhandling.Labware('A', 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware('B', 3, 4, min_volume=50, max_volume=250)
        wells = A.wells[:,:2]
        with evotools.Worklist() as worklist:
            worklist.transfer(A, wells, B, wells, 50)
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [150,150,200,200],
                [150,150,200,200],
                [150,150,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [50,50,0,0],
                [50,50,0,0],
                [50,50,0,0],
            ])))
            self.assertEqual(worklist, [
                # first transfer
                'A;A;;;1;;50.00;;;;',
                'D;B;;;1;;50.00;;;;',
                'W1;',
                'A;A;;;2;;50.00;;;;',
                'D;B;;;2;;50.00;;;;',
                'W1;',
                'A;A;;;3;;50.00;;;;',
                'D;B;;;3;;50.00;;;;',
                'W1;',
                'A;A;;;4;;50.00;;;;',
                'D;B;;;4;;50.00;;;;',
                'W1;',
                'A;A;;;5;;50.00;;;;',
                'D;B;;;5;;50.00;;;;',
                'W1;',
                'A;A;;;6;;50.00;;;;',
                'D;B;;;6;;50.00;;;;',
                'W1;',
            ])
            self.assertEqual(len(A.history), 2)
            self.assertEqual(len(B.history), 2)
        return
    
    def test_transfer_one_many(self):
        A = liquidhandling.Labware('A', 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware('B', 3, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, 'A01', B, ['B01', 'B02', 'B03'], 25)
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
            worklist.transfer(A, ['A01'], B, ['B01', 'B02', 'B03'], 25)
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [50,200,200,200],
                [200,200,200,200],
                [200,200,200,200],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [0,0,0,0],
                [50,50,50,0],
                [0,0,0,0],
            ])))
            self.assertEqual(worklist, [
                # first transfer
                'A;A;;;1;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;1;;25.00;;;;',
                'D;B;;;5;;25.00;;;;',
                'W1;',
                'A;A;;;1;;25.00;;;;',
                'D;B;;;8;;25.00;;;;',
                'W1;',
                # second transfer
                'A;A;;;1;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;1;;25.00;;;;',
                'D;B;;;5;;25.00;;;;',
                'W1;',
                'A;A;;;1;;25.00;;;;',
                'D;B;;;8;;25.00;;;;',
                'W1;',
            ])
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return
    
    def test_transfer_many_one(self):
        A = liquidhandling.Labware('A', 3, 4, min_volume=50, max_volume=250, initial_volumes=200)
        B = liquidhandling.Labware('B', 3, 4, min_volume=50, max_volume=250)
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
            self.assertEqual(worklist, [
                # first transfer
                'A;A;;;1;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;4;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;7;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
            ])
            self.assertEqual(len(A.history), 2)
            self.assertEqual(len(B.history), 2)
        return

    def test_tip_selection(self):
        A = liquidhandling.Labware('A', 3, 4, min_volume=10, max_volume=250, initial_volumes=100)
        with evotools.Worklist() as wl:
            wl.aspirate(A, 'A01', 10, tip=1)
            wl.aspirate(A, 'A01', 10, tip=2)
            wl.aspirate(A, 'A01', 10, tip=3)
            wl.aspirate(A, 'A01', 10, tip=4)
            wl.aspirate(A, 'A01', 10, tip=5)
            wl.aspirate(A, 'A01', 10, tip=6)
            wl.aspirate(A, 'A01', 10, tip=7)
            wl.aspirate(A, 'A01', 10, tip=8)
            wl.dispense(A, 'B01', 10, tip=evotools.Tip.T1)
            wl.dispense(A, 'B02', 10, tip=evotools.Tip.T2)
            wl.dispense(A, 'B03', 10, tip=evotools.Tip.T3)
            wl.dispense(A, 'B04', 10, tip=evotools.Tip.T4)
            wl.dispense(A, 'B04', 10, tip=evotools.Tip.T5)
            wl.dispense(A, 'B04', 10, tip=evotools.Tip.T6)
            wl.dispense(A, 'B04', 10, tip=evotools.Tip.T7)
            wl.dispense(A, 'B04', 10, tip=evotools.Tip.T8)
            self.assertEqual(wl, [
                'A;A;;;1;;10.00;;;1;',
                'A;A;;;1;;10.00;;;2;',
                'A;A;;;1;;10.00;;;4;',
                'A;A;;;1;;10.00;;;8;',
                'A;A;;;1;;10.00;;;16;',
                'A;A;;;1;;10.00;;;32;',
                'A;A;;;1;;10.00;;;64;',
                'A;A;;;1;;10.00;;;128;',
                'D;A;;;2;;10.00;;;1;',
                'D;A;;;5;;10.00;;;2;',
                'D;A;;;8;;10.00;;;4;',
                'D;A;;;11;;10.00;;;8;',
                'D;A;;;11;;10.00;;;16;',
                'D;A;;;11;;10.00;;;32;',
                'D;A;;;11;;10.00;;;64;',
                'D;A;;;11;;10.00;;;128;',
            ])
        return


class TestTroughLabwareWorklist(unittest.TestCase):
    def test_aspirate(self):
        source = liquidhandling.Labware('SourceLW', rows=1, columns=3, min_volume=10, max_volume=200, initial_volumes=200, virtual_rows=3)
        with evotools.Worklist() as wl:
            wl.aspirate(source, ['A01', 'A02', 'C02'], 50)
            wl.aspirate(source, ['A01', 'A02', 'C02'], [1,2,3])
            self.assertEqual(wl, [
                'A;SourceLW;;;1;;50.00;;;;',
                'A;SourceLW;;;4;;50.00;;;;',
                'A;SourceLW;;;6;;50.00;;;;',
                'A;SourceLW;;;1;;1.00;;;;',
                'A;SourceLW;;;4;;2.00;;;;',
                'A;SourceLW;;;6;;3.00;;;;',
            ])
            self.assertTrue(numpy.array_equal(source.volumes, [
                [149,95,200]
            ]))
            self.assertEqual(len(source.history), 3)
        return

    def test_dispense(self):
        destination = liquidhandling.Labware('DestinationLW', rows=1, columns=3, min_volume=10, max_volume=200, virtual_rows=3)
        with evotools.Worklist() as wl:
            wl.dispense(destination, ['A01', 'A02', 'A03', 'B01'], 50)
            wl.dispense(destination, ['A01', 'A02', 'C02'], [1,2,3])
            self.assertEqual(wl, [
                'D;DestinationLW;;;1;;50.00;;;;',
                'D;DestinationLW;;;4;;50.00;;;;',
                'D;DestinationLW;;;7;;50.00;;;;',
                'D;DestinationLW;;;2;;50.00;;;;',
                'D;DestinationLW;;;1;;1.00;;;;',
                'D;DestinationLW;;;4;;2.00;;;;',
                'D;DestinationLW;;;6;;3.00;;;;',
            ])
            self.assertTrue(numpy.array_equal(destination.volumes, [
                [101,55,50]
            ]))
            self.assertEqual(len(destination.history), 3)
        return

    def test_transfer_many_many(self):
        A = liquidhandling.Labware('A', 1, 4, min_volume=50, max_volume=2500, initial_volumes=2000, virtual_rows=3)
        B = liquidhandling.Labware('B', 3, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, ['A01', 'B01'], B, ['A01', 'B01'], 50)
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [1900,2000,2000,2000],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [50,0,0,0],
                [50,0,0,0],
                [0,0,0,0],
            ])))
            worklist.transfer(A, ['A03', 'B04'], B, ['A04', 'B04'], [50,75])
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [1900,2000,1950,1925],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [50,0,0,50],
                [50,0,0,75],
                [0,0,0,0],
            ])))
            self.assertEqual(worklist, [
                # first transfer
                'A;A;;;1;;50.00;;;;',
                'D;B;;;1;;50.00;;;;',
                'W1;',
                'A;A;;;2;;50.00;;;;',
                'D;B;;;2;;50.00;;;;',
                'W1;',
                # second transfer
                'A;A;;;7;;50.00;;;;',
                'D;B;;;10;;50.00;;;;',
                'W1;',
                'A;A;;;11;;75.00;;;;',
                'D;B;;;11;;75.00;;;;',
                'W1;',
            ])
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return
    
    def test_transfer_one_many(self):
        A = liquidhandling.Labware('A', 1, 4, min_volume=50, max_volume=2500, initial_volumes=2000, virtual_rows=3)
        B = liquidhandling.Labware('B', 3, 4, min_volume=50, max_volume=250)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, 'A01', B, ['B01', 'B02', 'B03'], 25)
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [1925,2000,2000,2000],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [0,0,0,0],
                [25,25,25,0],
                [0,0,0,0],
            ])))
            self.assertEqual(worklist, [
                # first transfer
                'A;A;;;1;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;1;;25.00;;;;',
                'D;B;;;5;;25.00;;;;',
                'W1;',
                'A;A;;;1;;25.00;;;;',
                'D;B;;;8;;25.00;;;;',
                'W1;',
            ])

            worklist.transfer(A, ['A01'], B, ['B01', 'B02', 'B03'], [25,30,35])
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [1835,2000,2000,2000],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [0,0,0,0],
                [50,55,60,0],
                [0,0,0,0],
            ])))
            self.assertEqual(worklist, [
                # first transfer
                'A;A;;;1;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;1;;25.00;;;;',
                'D;B;;;5;;25.00;;;;',
                'W1;',
                'A;A;;;1;;25.00;;;;',
                'D;B;;;8;;25.00;;;;',
                'W1;',
                # second transfer
                'A;A;;;1;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;1;;30.00;;;;',
                'D;B;;;5;;30.00;;;;',
                'W1;',
                'A;A;;;1;;35.00;;;;',
                'D;B;;;8;;35.00;;;;',
                'W1;',
            ])
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return
    
    def test_transfer_many_one(self):
        A = liquidhandling.Labware('A', 1, 4, min_volume=50, max_volume=2500, initial_volumes=[2000,1500,1000,500], virtual_rows=3)
        B = liquidhandling.Labware('B', 3, 4, min_volume=10, max_volume=250, initial_volumes=100)
        with evotools.Worklist() as worklist:
            worklist.transfer(A, ['A01', 'A02', 'A03'], B, 'B01', 25)
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [1975,1475,975,500],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [100,100,100,100],
                [175,100,100,100],
                [100,100,100,100],
            ])))
            self.assertEqual(worklist, [
                # first transfer
                'A;A;;;1;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;4;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;7;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
            ])

            worklist.transfer(B, B.wells[:,2], A, A.wells[:,3], [50,60,70])
            self.assertTrue(numpy.array_equal(A.volumes, numpy.array([
                [1975,1475,975,680],
            ])))
            self.assertTrue(numpy.array_equal(B.volumes, numpy.array([
                [100,100,50,100],
                [175,100,40,100],
                [100,100,30,100],
            ])))
            self.assertEqual(worklist, [
                # first transfer
                'A;A;;;1;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;4;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                'A;A;;;7;;25.00;;;;',
                'D;B;;;2;;25.00;;;;',
                'W1;',
                # second transfer
                'A;B;;;7;;50.00;;;;',
                'D;A;;;10;;50.00;;;;',
                'W1;',
                'A;B;;;8;;60.00;;;;',
                'D;A;;;11;;60.00;;;;',
                'W1;',
                'A;B;;;9;;70.00;;;;',
                'D;A;;;12;;70.00;;;;',
                'W1;',
            ])
            self.assertEqual(len(A.history), 3)
            self.assertEqual(len(B.history), 3)
        return


class TestLargeVolumeHandling(unittest.TestCase):
    def test_partition_volume_helper(self):
        self.assertEqual([], evotools._partition_volume(0, max_volume=950))
        self.assertEqual([550.3], evotools._partition_volume(550.3, max_volume=950))
        self.assertEqual([500, 500], evotools._partition_volume(1000, max_volume=950))
        self.assertEqual([500, 499], evotools._partition_volume(999, max_volume=950))
        self.assertEqual([667, 667, 666], evotools._partition_volume(2000, max_volume=950))
        return

    def test_worklist_constructor(self):
        with self.assertRaises(ValueError):
            with evotools.Worklist(max_volume=None) as wl:
                pass
        with evotools.Worklist(max_volume=800, auto_split=True) as wl:
            self.assertEqual(wl.max_volume, 800)
            self.assertEqual(wl.auto_split, True)
        with evotools.Worklist(max_volume=800, auto_split=False) as wl:
            self.assertEqual(wl.max_volume, 800)
            self.assertEqual(wl.auto_split, False)
        return

    def test_max_volume_checking(self):
        source = liquidhandling.Labware('WaterTrough', rows=1, columns=3, min_volume=1000, max_volume=100*1000, initial_volumes=50*1000, virtual_rows=3)
        destination = liquidhandling.Labware('WaterTrough', rows=1, columns=3, min_volume=1000, max_volume=100*1000, initial_volumes=50*1000, virtual_rows=3)
        with evotools.Worklist(max_volume=900, auto_split=False) as wl:
            with self.assertRaises(evotools.InvalidOperationError):
                wl.aspirate_well('WaterTrough', 1, 1000)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.dispense_well('WaterTrough', 1, 1000)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.aspirate(source, ['A01', 'A02', 'C02'], 1000)                
            with self.assertRaises(evotools.InvalidOperationError):
                wl.dispense(source, ['A01', 'A02', 'C02'], 1000)
            with self.assertRaises(evotools.InvalidOperationError):
                wl.transfer(source, ['A01', 'B01'], destination, ['A01', 'B01'], 1000)
        
        source = liquidhandling.Labware('WaterTrough', rows=1, columns=3, min_volume=1000, max_volume=100*1000, initial_volumes=50*1000, virtual_rows=3)
        destination = liquidhandling.Labware('WaterTrough', rows=1, columns=3, min_volume=1000, max_volume=100*1000, initial_volumes=50*1000, virtual_rows=3)
        with evotools.Worklist(max_volume=1200) as wl:
            wl.aspirate_well('WaterTrough', 1, 1000)
            wl.dispense_well('WaterTrough', 1, 1000)
            wl.aspirate(source, ['A01', 'A02', 'C02'], 1000)                
            wl.dispense(source, ['A01', 'A02', 'C02'], 1000)
            wl.transfer(source, ['A01', 'B01'], destination, ['A01', 'B01'], 1000)
        return

    def test_partition_by_columns_source(self):
        column_groups = evotools._partition_by_column(
            ['A01', 'B01', 'A03', 'B03', 'C02'],
            ['A01', 'B01', 'C01', 'D01', 'E01'],
            [2500, 3500, 1000, 500, 2000],
            partition_by='source'
        )
        self.assertEqual(len(column_groups), 3)
        self.assertEqual(column_groups[0], (
            ['A01', 'B01'],
            ['A01', 'B01'],
            [2500, 3500],
        ))
        self.assertEqual(column_groups[1], (
            ['C02'],
            ['E01'],
            [2000],
        ))
        self.assertEqual(column_groups[2], (
            ['A03', 'B03'],
            ['C01', 'D01'],
            [1000, 500],
        ))
        return

    def test_partition_by_columns_destination(self):
        column_groups = evotools._partition_by_column(
            ['A01', 'B01', 'A03', 'B03', 'C02'],
            ['A01', 'B01', 'C02', 'D01', 'E02'],
            [2500, 3500, 1000, 500, 2000],
            partition_by='destination'
        )
        self.assertEqual(len(column_groups), 2)
        self.assertEqual(column_groups[0], (
            ['A01', 'B01', 'B03'],
            ['A01', 'B01', 'D01'],
            [2500, 3500, 500],
        ))
        self.assertEqual(column_groups[1], (
            ['A03', 'C02'],
            ['C02', 'E02'],
            [1000, 2000],
        ))
        return

    def test_single_split(self):
        src = liquidhandling.Labware('A', 3, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = liquidhandling.Labware('B', 3, 2, min_volume=1000, max_volume=25000)
        with evotools.Worklist(auto_split=True) as wl:
            wl.transfer(
                src, 'A01',
                dst, 'A01',
                2000
            )
            self.assertEqual(wl, [
                'A;A;;;1;;667.00;;;;',
                'D;B;;;1;;667.00;;;;',
                'W1;',
                        # no breaks when pipetting single wells
                'A;A;;;1;;667.00;;;;',
                'D;B;;;1;;667.00;;;;',
                'W1;',
                        # no breaks when pipetting single wells
                'A;A;;;1;;666.00;;;;',
                'D;B;;;1;;666.00;;;;',
                'W1;',
                'B;',   # always break after partitioning
            ])
        self.assertTrue(numpy.array_equal(src.volumes, [
            [12000-2000, 12000],
            [12000, 12000],
            [12000, 12000],
        ]))
        self.assertTrue(numpy.array_equal(dst.volumes, [
            [2000, 0],
            [0, 0],
            [0, 0],
        ]))
        return

    def test_column_split(self):
        src = liquidhandling.Labware('A', 4, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = liquidhandling.Labware('B', 4, 2, min_volume=1000, max_volume=25000)
        with evotools.Worklist(auto_split=True) as wl:
            wl.transfer(
                src, ['A01', 'B01', 'D01', 'C01'],
                dst, ['A01', 'B01', 'D01', 'C01'],
                [1500, 250, 0, 1200]
            )
            self.assertEqual(wl, [
                'A;A;;;1;;750.00;;;;',
                'D;B;;;1;;750.00;;;;',
                'W1;',
                'A;A;;;2;;250.00;;;;',
                'D;B;;;2;;250.00;;;;',
                'W1;',
                        # D01 is ignored because the volume is 0
                'A;A;;;3;;600.00;;;;',
                'D;B;;;3;;600.00;;;;',
                'W1;',
                'B;',   # within-column break
                'A;A;;;1;;750.00;;;;',
                'D;B;;;1;;750.00;;;;',
                'W1;',
                'A;A;;;3;;600.00;;;;',
                'D;B;;;3;;600.00;;;;',
                'W1;',
                'B;',   # tailing break after partitioning
            ])
        self.assertTrue(numpy.array_equal(src.volumes, [
            [12000-1500, 12000],
            [12000-250, 12000],
            [12000-1200, 12000],
            [12000, 12000],
        ]))
        self.assertTrue(numpy.array_equal(dst.volumes, [
            [1500, 0],
            [250, 0],
            [1200, 0],
            [0, 0],
        ]))
        return

    def test_block_split(self):
        src = liquidhandling.Labware('A', 3, 2, min_volume=1000, max_volume=25000, initial_volumes=12000)
        dst = liquidhandling.Labware('B', 3, 2, min_volume=1000, max_volume=25000)
        with evotools.Worklist(auto_split=True) as wl:
            wl.transfer(
                     # A01, B01, A02, B02
                src, src.wells[:2,:],
                dst, ['A01', 'B01', 'C01', 'A02'],
                [1500, 250, 1200, 3000]
            )
            self.assertEqual(wl, [
                'A;A;;;1;;750.00;;;;',
                'D;B;;;1;;750.00;;;;',
                'W1;',
                'A;A;;;2;;250.00;;;;',
                'D;B;;;2;;250.00;;;;',
                'W1;',
                'B;',   # within-column 1 break
                'A;A;;;1;;750.00;;;;',
                'D;B;;;1;;750.00;;;;',
                'W1;',
                'B;',   # between-column 1/2 break
                'A;A;;;4;;600.00;;;;',
                'D;B;;;3;;600.00;;;;',
                'W1;',
                'A;A;;;5;;750.00;;;;',
                'D;B;;;4;;750.00;;;;',
                'W1;',
                'B;',   # within-column 2 break
                'A;A;;;4;;600.00;;;;',
                'D;B;;;3;;600.00;;;;',
                'W1;',
                'A;A;;;5;;750.00;;;;',
                'D;B;;;4;;750.00;;;;',
                'W1;',
                'B;',   # within-column 2 break
                'A;A;;;5;;750.00;;;;',
                'D;B;;;4;;750.00;;;;',
                'W1;',
                        # no break because only one well is accessed in this partition
                'A;A;;;5;;750.00;;;;',
                'D;B;;;4;;750.00;;;;',
                'W1;',
                'B;',   # tailing break after partitioning
            ])
        self.assertTrue(numpy.array_equal(src.volumes, [
            [12000-1500, 12000-1200],
            [12000-250, 12000-3000],
            [12000, 12000],
        ]))
        self.assertTrue(numpy.array_equal(dst.volumes, [
            [1500, 3000],
            [250, 0],
            [1200, 0],
        ]))
        return


if __name__ == '__main__':
    unittest.main()
