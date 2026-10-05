from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('tidy',Path(__file__).resolve().parents[1]/'scripts/tidy_fitlab.py')
tidy=importlib.util.module_from_spec(spec);spec.loader.exec_module(tidy)


class LayoutTests(unittest.TestCase):
    def test_move_references_backup_and_protected_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['docs','notebooks','src','tests','runs']:(root/name).mkdir()
            (root/'README.md').write_text('[scope](docs/UFM_FULL_SCOPE.md)')
            (root/'docs/UFM_FULL_SCOPE.md').write_text('Scope')
            (root/'docs/FITLAB.md').write_text('Fitlab guide')
            (root/'docs/old.before_patch.md').write_text('Original')
            (root/'docs/bundle.tar').write_bytes(b'bundle')
            (root/'src/model.py').write_text('protected')
            before=(root/'src/model.py').read_bytes()
            dry=tidy.organize(root)
            self.assertTrue(dry['moves']);self.assertFalse((root/'archive').exists())
            result=tidy.organize(root,True)
            self.assertEqual(result['protected_files_unchanged'],1)
            self.assertEqual((root/'README.md').read_text(),'[scope](docs/scope.md)')
            self.assertEqual((root/'src/model.py').read_bytes(),before)
            self.assertEqual((root/'archive/transfers/bundle.tar').read_bytes(),b'bundle')
            self.assertTrue((root/'archive/notes/old.before_patch.md').exists())
            self.assertIn('fitlab.md',[p.name for p in (root/'docs').iterdir()])
            with zipfile.ZipFile(root/'archive/layout-backup.zip') as z:
                self.assertIn(b'UFM_FULL_SCOPE.md',z.read('README.md'))
            self.assertEqual(json.loads((root/'archive/layout.json').read_text())['phase'],'complete')
            self.assertEqual(tidy.organize(root,True)['moves'],[])

    def test_collision_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'docs').mkdir();(root/'notebooks').mkdir()
            (root/'docs/UFM_FULL_SCOPE.md').write_text('original')
            (root/'docs/scope.md').write_text('keep')
            with self.assertRaises(FileExistsError):tidy.organize(root,True)
            self.assertEqual((root/'docs/UFM_FULL_SCOPE.md').read_text(),'original')
            self.assertFalse((root/'archive').exists())


if __name__=='__main__':unittest.main()
