"""CPU scalar golden values captured before integration edits, torch seed 123."""
import sys
from pathlib import Path
import torch
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/python'))
import Bose_Hubbard_NQS as bridge


class Baseline(unittest.TestCase):
    def test_legacy_scalar_gold(self):
        torch.manual_seed(123)
        model = bridge.build_model('FCNet', 3, 2, 4)
        configs = [[0, 1, 2], [2, 0, 1], [0, 0, 0]]
        self.assertEqual([bridge.predict_scalar(model, c) for c in configs], [
            0.3105006217956543, 0.308648943901062, 0.35193121433258057])
