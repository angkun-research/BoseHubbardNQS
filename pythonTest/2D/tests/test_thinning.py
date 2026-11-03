"""Exact seeded sampler compatibility and proposal/retention units."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate
import train
import vmc
from models import FCNet
from observables import sample_frozen


@torch.no_grad()
def pre_thinning_sampler_reference(model, initial_states, n_steps, burnin_steps, size):
    """Frozen original sampler control flow, independent of the new retain condition."""
    states = initial_states.clone()
    bonds = vmc._bond_tensor(model, states.device)
    current = vmc._log_amplitudes(model, states, size)
    samples = []
    counts = torch.zeros(3, dtype=torch.long)
    for step in range(burnin_steps + n_steps):
        proposed, valid = vmc.propose_hops(states, bonds, model.Nmax)
        new = vmc._log_amplitudes(model, proposed, size)
        accept = torch.rand(len(states)).log() < (2 * (new - current)).clamp(max=0)
        states = torch.where(accept[:, None], proposed, states)
        current = torch.where(accept, new, current)
        if step >= burnin_steps:
            counts += torch.stack((accept.sum(), (accept & valid).sum(), valid.sum()))
            samples.append(states.clone())
    accepted, moved, valid = counts.tolist()
    attempts = n_steps * len(states)
    info = {"attempts": attempts, "accepted": accepted, "moved": moved, "valid_proposals": valid,
            "accepted_rate": accepted / max(attempts, 1), "moved_rate": moved / max(attempts, 1),
            "valid_proposal_rate": valid / max(attempts, 1)}
    return torch.stack(samples).reshape(-1, model.n_sites), states, info


class ThinningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = FCNet(3, 2, Nmax=2, hidden_dim=4, pbc_x=False, pbc_y=True).double().eval()
        self.initial = torch.tensor([[2, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 2]])

    def test_default_and_explicit_one_match_frozen_old_seeded_behavior(self):
        torch.manual_seed(321)
        expected, walkers, old_info = pre_thinning_sampler_reference(self.model, self.initial, 6, 5, 2)
        for options in ({}, {"proposal_steps_per_sample": 1}):
            torch.manual_seed(321)
            actual, final, info = vmc.sample(self.model, self.initial, 6, 5, 2, **options)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(final, walkers, atol=0, rtol=0)
            for key, value in old_info.items():
                self.assertEqual(info[key], value)
            self.assertEqual(info["proposal_steps_per_sample"], 1)

    def test_every_kth_trajectory_final_walkers_and_all_proposal_counters(self):
        for stride in (2, 3, 5):
            with self.subTest(stride=stride):
                torch.manual_seed(987)
                full, walkers, reference_info = pre_thinning_sampler_reference(self.model, self.initial, 5 * stride, 7, 2)
                torch.manual_seed(987)
                thinned, final, info = vmc.sample(self.model, self.initial, 5, 7, 2, proposal_steps_per_sample=stride)
                expected = full.reshape(5 * stride, 3, 6)[stride - 1::stride]
                torch.testing.assert_close(thinned.reshape(5, 3, 6), expected, atol=0, rtol=0)
                torch.testing.assert_close(final, walkers, atol=0, rtol=0)
                for key, value in reference_info.items():
                    self.assertEqual(info[key], value)
                self.assertTrue(torch.all(thinned.sum(1) == 2))
                self.assertTrue(torch.all((thinned >= 0) & (thinned <= 2)))
                self.assertEqual(info["retained_steps_per_walker"], 5)
                self.assertEqual(info["postburn_proposal_steps_per_walker"], 5 * stride)
                self.assertEqual(info["burnin_proposal_steps_per_walker"], 7)
                self.assertEqual(info["attempts"], 5 * stride * 3)
                self.assertIn("not a lattice sweep", info["attempts_unit"])

    def test_frozen_wrapper_retains_time_axes_at_stride_and_empty_retention(self):
        torch.manual_seed(12)
        full, walkers, _ = vmc.sample(self.model, self.initial, 12, 2, 2)
        torch.manual_seed(12)
        chains, final, info = sample_frozen(self.model, self.initial, 4, 2, 2, proposal_steps_per_sample=3)
        torch.testing.assert_close(chains, full.reshape(12, 3, 6)[2::3], atol=0, rtol=0)
        torch.testing.assert_close(final, walkers, atol=0, rtol=0)
        self.assertEqual(info["attempts"], 36)
        empty, _, info = sample_frozen(self.model, self.initial, 0, 2, 2, proposal_steps_per_sample=3)
        self.assertEqual(empty.shape, (0, 3, 6))
        self.assertEqual(info["attempts"], 0)

    def test_invalid_thin_rejected_early_and_cli_default_unchanged(self):
        with patch.object(self.model, "forward", side_effect=AssertionError("forward called")):
            for value in (0, -1, True, 1.0, float("nan"), "2", None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    vmc.sample(self.model, self.initial, 1, proposal_steps_per_sample=value)
        self.assertEqual(train.parse_args(["--device", "cpu"]).thin, 1)
        self.assertEqual(train.parse_args(["--device", "cpu", "--thin", "7"]).thin, 7)
        with patch.object(evaluate, "load_model", side_effect=AssertionError("model loaded")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate.main(["--checkpoint", "does-not-exist.pt", "--thin", "0"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.parse_args(["--device", "cpu", "--thin", "0"])


if __name__ == "__main__":
    unittest.main()
