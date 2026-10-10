"""Transport fixtures test archive handling, not checkpoint model quality."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('collector', Path(__file__).resolve().parents[1]/'scripts/collect_colab_study.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def make_archive(self, root, state='complete'):
        job = root/'source/binary-42'
        files = {'status.json':json.dumps({'state':state}), 'evaluation.json':'{}',
                 'training/config.json':'{}', 'training/train-puzzles.json':'[]',
                 'training/holdout.json':'[]', 'training/wandb-run.json':'{}',
                 'training/adapter/adapter_config.json':'{}',
                 'training/adapter/adapter_model.safetensors':'transport fixture, not weights',
                 'training/metrics.jsonl': ''.join(json.dumps({'step':i,'reward':0})+'\n' for i in range(1,301))+json.dumps({'train_runtime':100})+'\n'}
        for name,text in files.items():
            path = job/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text)
        archive = root/'job.tar.gz'
        with tarfile.open(archive,'w:gz') as handle:
            handle.add(job,arcname=job.name)
        return archive, hashlib.sha256(archive.read_bytes()).hexdigest()

    def test_verified_archive_installs_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive,checksum = self.make_archive(root)
            destination = root/'collected/binary-42'
            collector.install_archive(archive,checksum,destination)
            self.assertTrue((destination/'training/adapter/adapter_model.safetensors').is_file())
            self.assertEqual(json.loads((destination/'collection.json').read_text())['sha256'],checksum)
            with self.assertRaises(FileExistsError):
                collector.install_archive(archive,checksum,destination)

    def test_corrupt_and_unfinished_archives_are_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive,checksum = self.make_archive(root,state='running')
            destination = root/'collected/binary-42'
            with self.assertRaisesRegex(ValueError,'checksum'):
                collector.install_archive(archive,'0'*64,destination)
            self.assertFalse(destination.exists())
            with self.assertRaisesRegex(ValueError,'not complete'):
                collector.install_archive(archive,checksum,destination)
            self.assertFalse(destination.exists())
