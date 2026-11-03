"""Independent combinatorics and brute-force projection/gauge checks."""
import itertools
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import leakage_analysis as leakage
import exact_diagonalization as ed


class ConstantState(nn.Module):
    def __init__(self,n_sites=3,Nmax=2,value=.7):
        super().__init__();self.n_sites=n_sites;self.Nmax=Nmax
        self.constant=nn.Parameter(torch.tensor(value,dtype=torch.float64))
    def forward(self,x):return self.constant.expand(len(x))


class TwoSiteState(ConstantState):
    def __init__(self,Nmax=2):super().__init__(2,Nmax)
    def forward(self,x):
        n=(x*torch.arange(self.Nmax+1,dtype=x.dtype)).sum(-1)
        return self.constant+.3*n[:,0]-.2*n[:,1]+.1*n[:,0]*n[:,1]


class LeakageTests(unittest.TestCase):
    def test_constant_degeneracies_mean_weight_and_window_not_global(self):
        model=ConstantState()
        full=leakage.analyze_sectors(model,2,window=[1,2,3],batch_size=4)
        counts=[1,3,6,7,6,3,1]  # coefficients of(1+x+x²)^3
        for n,row in enumerate(full['sectors']):
            self.assertEqual(row['dimension'],counts[n])
            self.assertAlmostEqual(row['logZ_N'],math.log(counts[n])+1.4,places=13)
            self.assertAlmostEqual(row['log_mean_squared_raw_amplitude_per_configuration'],1.4,places=13)
            self.assertAlmostEqual(row['P_full'],counts[n]/27,places=14)
        window=leakage.analyze_sectors(model,2,[1,2,3],batch_size=2)
        self.assertNotIn('P_full_N0',window)
        self.assertNotIn('leakage_full',window)
        self.assertTrue(all('P_full' not in row for row in window['sectors']))
        self.assertAlmostEqual(window['window_conditional_P_N0'],6/16,places=14)
        self.assertAlmostEqual(full['P_full_N0'],6/27,places=14)
        self.assertAlmostEqual(full['window_mass_in_full'],16/27,places=14)
        self.assertAlmostEqual(full['leakage_full'],1-6/27,places=14)

    def test_two_site_direct_brute_force_and_batch_invariance(self):
        model=TwoSiteState()
        states=list(itertools.product(range(3),repeat=2))
        f=np.array([.7+.3*a-.2*b+.1*a*b for a,b in states])
        expected={n:np.log(np.exp(2*f[[a+b==n for a,b in states]]).sum()) for n in range(5)}
        for batch in (1,4,16):
            result=leakage.analyze_sectors(model,2,batch_size=batch)
            for row in result['sectors']:self.assertAlmostEqual(row['logZ_N'],expected[row['N']],places=13)
        stream,counts=leakage.state_batches(2,2,batch_size=4)
        np.testing.assert_array_equal(np.concatenate(list(stream)),np.array(states))
        stream,_=leakage.state_batches(2,2,[2],batch_size=2)
        np.testing.assert_array_equal(np.concatenate(list(stream)),np.array([[0,2],[1,1],[2,0]]))
        self.assertEqual(sum(counts.values()),9)

    def test_exact_projection_and_full_overlap_energy_identity(self):
        model=TwoSiteState(Nmax=1)
        states=np.array(list(itertools.product(range(2),repeat=2)))
        f=leakage.evaluate_logs(model,states)
        raw=np.exp(f);raw/=np.linalg.norm(raw)
        mask=states.sum(1)==1
        canonical=raw[mask]/np.linalg.norm(raw[mask])
        target=np.ones(2)/math.sqrt(2)
        F0=float(abs(canonical@target))
        target_full=np.array([0,1/math.sqrt(2),1/math.sqrt(2),0])
        full=leakage.analyze_sectors(model,1,batch_size=3)
        self.assertAlmostEqual(leakage.full_overlap(F0,full['P_full_N0']),abs(raw@target_full),places=14)
        projected=raw.copy();projected[~mask]=0;projected/=np.linalg.norm(projected)
        np.testing.assert_allclose(projected[mask],canonical,atol=1e-15)
        np.testing.assert_array_equal(projected[~mask],0)
        H=np.array([[0.,-1.],[-1.,0.]])  # one hardcore hopping bond
        self.assertAlmostEqual(projected[mask]@H@projected[mask],canonical@H@canonical,places=14)
        self.assertAlmostEqual(abs(projected@target_full),F0,places=14)
        self.assertEqual(leakage.projected_sector_probabilities(range(3),1),{0:0.,1:1.,2:0.})

    def test_linear_number_gauge_unidentifiability_and_its_limit(self):
        model=TwoSiteState()
        original=leakage.analyze_sectors(model,2,batch_size=4)
        original_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
        class Gauged(nn.Module):
            def __init__(self,base,a):
                super().__init__();self.base=base;self.n_sites=base.n_sites;self.Nmax=base.Nmax;self.a=a
            def forward(self,x):
                n=(x*torch.arange(self.Nmax+1,dtype=x.dtype)).sum((1,2))
                return self.base(x)+self.a*(n-2)
        canonical_states=np.array([[0,2],[1,1],[2,0]])
        for a in (-4.,0.,4.):
            analytic=leakage.apply_number_gauge(original,a)
            actual=leakage.analyze_sectors(Gauged(model,a),2,batch_size=4)
            for x,y in zip(analytic['sectors'],actual['sectors']):
                self.assertAlmostEqual(x['logZ_N'],y['logZ_N'],places=12)
                self.assertAlmostEqual(x['P_full'],y['P_full'],places=13)
            np.testing.assert_array_equal(leakage.evaluate_logs(Gauged(model,a),canonical_states),leakage.evaluate_logs(model,canonical_states))
            self.assertEqual(np.mean((canonical_states.sum(1)-2)**2),0)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,original_weights[k]))
        # A linear tilt can move relative sector masses arbitrarily, but cannot
        # independently set all sectors or make canonicalP arbitrarily close1.
        constant=leakage.analyze_sectors(ConstantState(2,1,0),1)
        for a in (-4.,0.,4.):
            self.assertLessEqual(leakage.apply_number_gauge(constant,a)['P_full_N0'],.5+1e-14)
        self.assertLess(leakage.apply_number_gauge(constant,4)['P_full_N0'],.001)

    def test_guard_order_finite_integer_outputs_no_H_or_ED(self):
        model=ConstantState(9,4)
        with patch.object(model,'forward',side_effect=AssertionError('forward before guard')):
            for options in ({'max_states':10},{'sectors':[9],'max_states':10},{'sectors':[7,8],'window':[9]}):
                with self.assertRaises(ValueError):leakage.analyze_sectors(model,9,**options)
        with self.assertRaisesRegex(ValueError,'integer'):leakage.evaluate_logs(TwoSiteState(),[[.5,1.]])
        with self.assertRaises(ValueError):leakage.state_batches(2,1,[1,1])
        with self.assertRaises(ValueError):leakage.state_batches(2,1,[3])
        with self.assertRaises(ValueError):leakage.state_batches(2,1,batch_size=0)
        with patch.object(ed,'exact_ground_state',side_effect=AssertionError('unexpected ED')), \
             patch.object(ed,'build_bose_hubbard_hamiltonian',side_effect=AssertionError('unexpected H')):
            result=leakage.analyze_sectors(TwoSiteState(),2)
            self.assertEqual(result['enumerated_state_count'],9)
        with self.assertRaises(ValueError):leakage.analyze_sectors(ConstantState(2,2,float('inf')),2)
        with self.assertRaises(ValueError):leakage.full_overlap(1.1,.5)
        zero=leakage.analyze_sectors(ConstantState(1,0),0)
        self.assertEqual(zero['P_full_N0'],1)
        self.assertEqual(zero['leakage_full'],0)


if __name__=='__main__':unittest.main()
