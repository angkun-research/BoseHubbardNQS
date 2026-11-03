"""Independent small operator references and covariance-sensitive composites."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'benchmarks'))
from exact_diagonalization import build_bose_hubbard_hamiltonian
from periodic_observables import physical_bonds,sparse_bond_operators,check_operator_H,state_observables,frozen_observables
from run_leakage_analysis import selection_settings


class PeriodicBenchmarkTests(unittest.TestCase):
    def test_torus_bond_order_counts_and_literal_rectangle(self):
        bonds=physical_bonds(2,3)
        self.assertEqual([(b['i'],b['j']) for b in bonds[:3]],[(0,1),(2,3),(4,5)])
        self.assertEqual({(b['i'],b['j']) for b in bonds[3:]},{(0,2),(0,4),(2,4),(1,3),(1,5),(3,5)})
        self.assertEqual([b['axis'] for b in bonds],['x']*3+['y']*6)
        square=physical_bonds(3,3)
        self.assertEqual(len(square),18)
        self.assertEqual([b['axis'] for b in square],['x']*9+['y']*9)
        self.assertEqual(len({(b['i'],b['j']) for b in square}),18)

    def test_two_site_nonuniform_exact_ladder_and_sparse_operator(self):
        H,basis=build_bose_hubbard_hamiltonian(2,1,1,1,t=1.,U=5.4,mu=0.)
        basis=np.array(basis);bonds=physical_bonds(2,1)
        ops=sparse_bond_operators(basis,1,bonds)
        self.assertEqual(check_operator_H(H,basis,ops,1.,5.4,0.),0.)
        result=state_observables(np.array([1.,2.]),basis,H,bonds,Nmax=1,U=5.4,operators=ops)
        np.testing.assert_allclose(result['density'],[.8,.2],atol=1e-15)
        np.testing.assert_allclose(result['rho'],[[.8,.4],[.4,.2]],atol=1e-15)
        self.assertAlmostEqual(result['Gx'],.4)
        self.assertIsNone(result['Gy'])
        self.assertAlmostEqual(result['energy'],-.8)
        self.assertEqual(result['meanD'],0)

    def test_single_particle_uniform_square_independent_spectrum(self):
        H,basis=build_bose_hubbard_hamiltonian(2,2,1,1,t=1.,U=3.2,mu=.3)
        basis=np.array(basis);bonds=physical_bonds(2,2);ops=sparse_bond_operators(basis,1,bonds)
        self.assertLess(check_operator_H(H,basis,ops,1.,3.2,.3),1e-14)
        result=state_observables(np.ones(4),basis,H,bonds,Nmax=1,U=3.2,mu=.3,operators=ops)
        np.testing.assert_allclose(result['rho'],np.full((4,4),.25),atol=1e-15)
        self.assertAlmostEqual(result['energy'],-2.3)
        self.assertAlmostEqual(result['variance'],0)
        self.assertAlmostEqual(result['kinetic'],-2)

    def test_composites_keep_bond_and_site_covariance(self):
        bonds=physical_bonds(2,3);T,C,N=4,2,6
        fluct=np.array([[.1,-.1],[-.1,.1],[.1,-.1],[-.1,.1]])
        # Pair/block patterns vary while Gx and meanD cancel exactly samplewise.
        fluct=np.array([[.1,.1],[.1,.1],[-.1,-.1],[-.1,-.1]])
        rho=np.ones((T,C,N,N));density=np.ones((T,C,N));moment=np.ones((T,C,N))
        for k,b in enumerate(bonds):
            value=np.ones((T,C))
            if k==0:value=value+fluct
            if k==1:value=value-fluct
            rho[...,b['i'],b['j']]=rho[...,b['j'],b['i']]=value
        moment[...,0]+=fluct;moment[...,1]-=fluct
        energy=np.full((T,C),-18+16.74/2*6)
        raw={'density':density,'onsite_factorial_moment':moment,'local_energy':energy,
             'physical_pairs':np.array([(i,j) for i in range(N) for j in range(N)]),
             'one_body_estimators':rho.reshape(T,C,N*N)}
        result=frozen_observables(raw,bonds,block_length=2)
        self.assertGreater(result['statistics']['bond_G']['conditional_sem'][0],0)
        self.assertGreater(result['statistics']['onsite_factorial']['conditional_sem'][0],0)
        self.assertAlmostEqual(result['statistics']['Gx']['conditional_sem'],0)
        self.assertAlmostEqual(result['statistics']['meanD']['conditional_sem'],0)
        self.assertAlmostEqual(result['statistics']['kinetic']['conditional_sem'],0)
        self.assertAlmostEqual(result['observed']['Gx'],1)
        self.assertAlmostEqual(result['observed']['meanD'],1)
        self.assertLess(result['checks']['perconfiguration_energy_reconstruction_max_abs'],1e-12)

    def test_leakage_selection_honors_axes_and_preserves_cylinder_default(self):
        def selection(px,py):
            m={'Lx':3,'Ly':3,'Ntot':9,'Nmax':4,'pbc_x':px,'pbc_y':py,'t':1.,'U':16.74,'mu':0.}
            return {'models':[{'metadata':m}]}
        cylinder=selection_settings(selection(False,True))
        self.assertFalse(cylinder['pbc_x']);self.assertTrue(cylinder['pbc_y'])
        periodic=selection(True,True);periodic.update(geometry=[3,3],N0=9,Nmax=4,pbc_x=True,pbc_y=True)
        self.assertTrue(selection_settings(periodic)['pbc_x'])
        periodic['pbc_x']=False
        with self.assertRaisesRegex(ValueError,'match'):selection_settings(periodic)


if __name__=='__main__':unittest.main()
