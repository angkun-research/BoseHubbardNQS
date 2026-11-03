"""Independent cylinder group action for ordered Jastrow reflection orbits."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys,tempfile,unittest
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models import JastrowNet,lift_jastrow_reflection_orbit,save_model,load_model,parameter_count
from vmc import generate_initial_states,compute_log_psi_jacobian,sample
import train,train_ed


def group_reference(model,states):
    # Independently expand representative ordered interactions under the physical
    # group: x reflection, periodic y translations, y reflection. Average all
    # orientations reaching an unordered physical pair (stabilizer projection).
    potentials={}
    for k,(left,right,dy) in enumerate(model.pair_orbits.tolist()):
        images={}
        for rx in (False,True):
            for ry in (False,True):
                for shift in range(model.Ly):
                    xa,xb=(model.Lx-1-left,model.Lx-1-right) if rx else (left,right)
                    ya,yb=shift,((-dy if ry else dy)+shift)%model.Ly
                    i,j=ya*model.Lx+xa,yb*model.Lx+xb
                    key=(min(i,j),max(i,j))
                    images.setdefault(key,[]).append(model.pair[k] if i<j else model.pair[k].T)
        for key,tables in images.items():
            if key in potentials:raise AssertionError('distinct representative classes overlap')
            potentials[key]=torch.stack(tables).mean(0)
    assert len(potentials)==model.n_sites*(model.n_sites-1)//2
    result=[]
    for state in states:
        value=sum(model.onsite[min(i%model.Lx,model.Lx-1-i%model.Lx),n] for i,n in enumerate(state))
        for (i,j),table in potentials.items():value=value+table[state[i],state[j]]
        result.append(value)
    return torch.stack(result)


class ReflectionOrbitTests(unittest.TestCase):
    def test_random_tables_against_independent_group_expansion(self):
        for Lx,Ly in ((3,4),(4,3),(4,4)):
            torch.manual_seed(57)
            model=JastrowNet(Lx,Ly,Nmax=2,pbc_x=False,pbc_y=True,pair_sharing='reflection_orbit').double()
            with torch.no_grad():
                model.onsite.normal_();model.pair.normal_()
            states=generate_initial_states(Lx*Ly,Lx*Ly,2,7)
            actual=model(F.one_hot(states,3).double())
            torch.testing.assert_close(actual,group_reference(model,states),rtol=1e-12,atol=1e-11)

    def test_physical_group_invariance_with_asymmetric_occupation_tables(self):
        torch.manual_seed(41)
        model=JastrowNet(4,3,Nmax=2,pbc_x=False,pbc_y=True,pair_sharing='reflection_orbit').double()
        with torch.no_grad():
            model.onsite.normal_();model.pair.normal_()
        states=generate_initial_states(12,12,2,9);grid=states.reshape(-1,3,4)
        expected=model(F.one_hot(states,3).double())
        for transformed in (grid.flip(-1),grid.flip(-2),grid.roll(1,-2)):
            torch.testing.assert_close(model(F.one_hot(transformed.reshape(-1,12),3).double()),expected,rtol=1e-12,atol=1e-11)

    def test_edge_bulk_order_is_not_unconditionally_symmetrized(self):
        model=JastrowNet(4,1,Nmax=1,pbc_x=False,pbc_y=True,pair_sharing='reflection_orbit').double()
        k=model.pair_orbits.tolist().index([0,1,0])
        with torch.no_grad():model.pair[k,1,0]=1;model.pair[k,0,1]=-1
        states=torch.tensor([[1,0,0,0],[0,0,0,1],[0,1,0,0]])
        torch.testing.assert_close(model(F.one_hot(states,2).double()),torch.tensor([1.,1.,-1.],dtype=torch.float64),rtol=0,atol=0)
        self.assertFalse(model.symmetric_classes[k])
        self.assertTrue(model.symmetric_classes[model.pair_orbits.tolist().index([0,3,0])])
        same=JastrowNet(4,3,Nmax=1,pbc_x=False,pbc_y=True,pair_sharing='reflection_orbit')
        self.assertTrue(same.symmetric_classes[same.pair_orbits.tolist().index([0,0,1])])

    def test_exact_lift_feasible_and_sampled_legacy_cylinders(self):
        for Lx,Ly in ((3,3),(4,4),(6,4),(1,1)):
            torch.manual_seed(35)
            old=JastrowNet(Lx,Ly,Nmax=2,pbc_x=False,pbc_y=True).double()
            with torch.no_grad():
                old.onsite.normal_(std=.03);old.pair.normal_(std=.03)
            before={n:v.clone() for n,v in old.state_dict().items()}
            new=lift_jastrow_reflection_orbit(old)
            states=generate_initial_states(Lx*Ly,Lx*Ly,2,13)
            sampled,_,_=sample(old,states[:2],3,2)
            x=F.one_hot(torch.cat((states,sampled)),3).double()
            torch.testing.assert_close(old(x),new(x),rtol=1e-12,atol=1e-11)
            self.assertTrue(all(torch.equal(v,before[n]) for n,v in old.state_dict().items()))
        for Lx,expected in ((3,160),(4,410),(6,840)):
            self.assertEqual(parameter_count(JastrowNet(Lx,3 if Lx==3 else 4,pbc_x=False,pbc_y=True,pair_sharing='reflection_orbit')),expected)

    def test_new_checkpoint_and_missing_legacy_mode_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='test-Jorbit-') as folder:
            old=JastrowNet(4,3,Nmax=2,pbc_x=False,pbc_y=True).double()
            with torch.no_grad():old.onsite.normal_();old.pair.normal_()
            legacy=Path(folder)/'legacy.pt';newpath=Path(folder)/'orbit.pt'
            save_model(legacy,old,12)
            raw=torch.load(legacy,weights_only=True);raw['metadata']['architecture_settings'].pop('pair_sharing');torch.save(raw,legacy)
            loaded,meta=load_model(legacy)
            self.assertEqual(meta['architecture_settings']['pair_sharing'],'displacement')
            states=generate_initial_states(12,12,2,9);x=F.one_hot(states,3).double()
            torch.testing.assert_close(old(x),loaded(x),rtol=0,atol=0)
            new=lift_jastrow_reflection_orbit(old);save_model(newpath,new,12)
            saved,meta=load_model(newpath)
            self.assertEqual(meta['architecture_settings'],{'pair_sharing':'reflection_orbit'})
            torch.testing.assert_close(saved(x),new(x),rtol=0,atol=0)

    def test_jacobians_and_early_cylinder_rejection(self):
        model=JastrowNet(3,2,Nmax=2,pbc_x=False,pbc_y=True,pair_sharing='reflection_orbit').double()
        states=generate_initial_states(6,6,2,5)
        expected=compute_log_psi_jacobian(model,states)
        for method in ('vjp','vmap'):
            torch.testing.assert_close(compute_log_psi_jacobian(model,states,jacobian_method=method,jacobian_batch_size=4),expected,rtol=1e-12,atol=1e-12)
        for px,py in ((True,True),(True,False),(False,False)):
            with self.assertRaises(ValueError):JastrowNet(3,3,pbc_x=px,pbc_y=py,pair_sharing='reflection_orbit')
        for module in (train,train_ed):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):module.parse_args(['--arch','JastrowNet','--pair-sharing','reflection_orbit'])
            args=module.parse_args(['--arch','JastrowNet','--pair-sharing','reflection_orbit','--no-pbc-x','--pbc-y'])
            self.assertEqual(args.pair_sharing,'reflection_orbit')


if __name__=='__main__':unittest.main()
