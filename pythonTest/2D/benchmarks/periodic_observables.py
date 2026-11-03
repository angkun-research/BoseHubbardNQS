"""Exact row-ladder/sparse-operator and stored-chain observable benchmarks.

rho[i,j]=<b_i^dag b_j>; density is <n_i>. D is <n_i(n_i-1)>, not generally
Var(n_i). Bonds are unique unordered edges, ordered x then y. No training.
"""
import numpy as np
import scipy.sparse as sp
import torch

from lattice import bonds_2d
from observables import block_mean_statistics


def physical_bonds(Lx,Ly,pbc_x=True,pbc_y=True):
    rows=[]
    for i,j in bonds_2d(Lx,Ly,pbc_x=pbc_x,pbc_y=pbc_y):
        axis='x' if i//Lx==j//Lx else 'y'
        if axis=='y' and i%Lx!=j%Lx:raise ValueError('non-axis bond')
        rows.append({'i':i,'j':j,'axis':axis,'xy_i':[i%Lx,i//Lx],'xy_j':[j%Lx,j//Lx]})
    return [dict(index=k,**row) for k,row in enumerate(sorted(rows,key=lambda b:(b['axis'],b['i'],b['j'])))]


def _codes(basis,Nmax):
    basis=np.asarray(basis,dtype=np.int64)
    strides=np.array([(Nmax+1)**i for i in range(basis.shape[1]-1,-1,-1)],dtype=np.int64)
    codes=basis@strides
    if np.any(np.diff(codes)<=0):raise ValueError('canonical basis must be unique lexicographic order')
    return basis,strides,codes


def sparse_bond_operators(basis,Nmax,bonds):
    """Independent ket-action O_b=(b_i^dag b_j+b_j^dag b_i)/2 in canonicalbasis."""
    basis,strides,codes=_codes(basis,Nmax);D=len(basis);operators=[]
    for bond in bonds:
        rows=[];columns=[];values=[]
        for dst,src in ((bond['i'],bond['j']),(bond['j'],bond['i'])):
            source=np.flatnonzero((basis[:,src]>0)&(basis[:,dst]<Nmax))
            target_code=codes[source]+strides[dst]-strides[src]
            target=np.searchsorted(codes,target_code)
            if np.any(target>=D) or not np.array_equal(codes[target],target_code):raise ValueError('basis not closed under hopping')
            rows.extend(target.tolist());columns.extend(source.tolist())
            values.extend((.5*np.sqrt((basis[source,dst]+1)*basis[source,src])).tolist())
        operators.append(sp.coo_matrix((values,(rows,columns)),shape=(D,D)).tocsr())
    return operators


def check_operator_H(H,basis,operators,t,U,mu):
    diagonal=.5*U*(basis*(basis-1)).sum(1)-mu*basis.sum(1)
    reference=sp.diags(diagonal,format='csr')
    for op in operators:reference=reference-2*t*op
    difference=reference-H
    error=float(abs(difference.data).max()) if difference.nnz else 0.
    if error>1e-12:raise RuntimeError('independent sparse ladder H disagrees with ED H')
    return error


def _bond_means(rho,bonds):
    values=np.array([.5*(rho[b['i'],b['j']]+rho[b['j'],b['i']]) for b in bonds])
    direction={a:float(values[[b['axis']==a for b in bonds]].mean()) if any(b['axis']==a for b in bonds) else None for a in ('x','y')}
    return values,direction


def state_observables(psi,basis,H,bonds,*,Nmax=4,t=1.,U=16.74,mu=0.,operators=None):
    """Exact normalized sector expectations; row-action independent of sampler."""
    if np.iscomplexobj(psi):raise ValueError('this reference supports real states')
    psi=np.asarray(psi,dtype=np.float64);basis,strides,codes=_codes(basis,Nmax)
    if psi.shape!=(len(basis),) or not np.isfinite(psi).all() or np.linalg.norm(psi)==0:raise ValueError('invalid state')
    psi=psi/np.linalg.norm(psi);p=psi**2;N=basis.shape[1]
    density=p@basis;moment=p@(basis*(basis-1));rho=np.zeros((N,N))
    for i in range(N):
        rho[i,i]=density[i]
        for j in range(N):
            if i==j:continue
            source=np.flatnonzero((basis[:,i]>0)&(basis[:,j]<Nmax))
            connected=codes[source]-strides[i]+strides[j]
            target=np.searchsorted(codes,connected)
            if np.any(target>=len(basis)) or not np.array_equal(codes[target],connected):raise ValueError('basis not closed')
            rho[i,j]=np.sum(psi[source]*psi[target]*np.sqrt(basis[source,i]*(basis[source,j]+1)))
    applied=H@psi;energy=float(psi@applied)
    G,direction=_bond_means(rho,bonds)
    kinetic=float(-2*t*G.sum());potential=float(.5*U*moment.sum())
    reconstructed=kinetic+potential-mu*density.sum()
    if abs(reconstructed-energy)>1e-10:raise RuntimeError('NN/onsite energy reconstruction failed')
    sparse_error=None
    if operators is not None:
        independent=np.array([psi@(op@psi) for op in operators])
        sparse_error=float(abs(independent-G).max()) if len(G) else 0.
        if sparse_error>1e-12:raise RuntimeError('row-ladder rho disagrees with sparse ket bond operators')
    counts={a:sum(b['axis']==a for b in bonds) for a in ('x','y')}
    by_direction=-2*t*sum(counts[a]*direction[a] for a in ('x','y') if counts[a])
    return {'density':density.tolist(),'onsite_factorial':moment.tolist(),'meanD':float(moment.mean()),
            'rho':rho.tolist(),'bond_G':G.tolist(),'Gx':direction['x'],'Gy':direction['y'],
            'energy':energy,'variance':float(np.linalg.norm(applied-energy*psi)**2),
            'kinetic':kinetic,'potential':potential,'energy_reconstructed':float(reconstructed),
            'kinetic_from_direction_means':float(by_direction),
            'checks':{'rho_trace':float(np.trace(rho)),'rho_diagonal_minus_density_max_abs':float(abs(np.diag(rho)-density).max()),
                      'rho_raw_hermiticity_max_abs':float(abs(rho-rho.T).max()),
                      'energy_reconstruction_abs':float(abs(reconstructed-energy)),
                      'sparse_operator_G_max_abs_difference':sparse_error}}


def residuals(observed,reference):
    out={}
    for key in ('density','onsite_factorial','rho','bond_G'):
        out[key]=(np.asarray(observed[key])-reference[key]).tolist()
    for key in ('meanD','Gx','Gy','energy','kinetic','potential'):
        out[key]=None if observed.get(key) is None else float(observed[key]-reference[key])
    out['max_abs_density']=float(abs(np.asarray(out['density'])).max())
    out['max_abs_bond_G']=float(abs(np.asarray(out['bond_G'])).max()) if out['bond_G'] else 0.
    return out


def _stats(values,block_length):
    result=block_mean_statistics(torch.as_tensor(values),block_length)
    errors=[result[k] for k in ('block_standard_error','between_chain_standard_error') if result[k] is not None]
    sem=torch.stack(errors).max(0).values if errors else None
    def value(x):return x.tolist() if isinstance(x,torch.Tensor) else x
    return {**{key:value(x) for key,x in result.items() if key!='block_means'},
            'conditional_sem':value(sem)}


def frozen_observables(raw,bonds,*,block_length=64,t=1.,U=16.74,mu=0.):
    """Composite each(time,walker)sample BEFOREblocking: keep all covariance."""
    density=np.asarray(raw['density'],dtype=np.float64);moment=np.asarray(raw['onsite_factorial_moment'],dtype=np.float64)
    if density.ndim!=3 or moment.shape!=density.shape:raise ValueError('time/walker/site axes required')
    T,C,N=density.shape
    pairs=[tuple(map(int,p)) for p in raw['physical_pairs']]
    if len(pairs)!=N*N or set(pairs)!={(i,j) for i in range(N) for j in range(N)}:raise ValueError('full physical rho pairs required')
    pair_samples=np.asarray(raw['one_body_estimators'],dtype=np.float64)
    if pair_samples.shape!=(T,C,N*N):raise ValueError('pair sample axes mismatch')
    rho=np.empty((T,C,N,N),dtype=np.float64)
    for k,(i,j) in enumerate(pairs):rho[...,i,j]=pair_samples[...,k]
    G=np.stack([.5*(rho[...,b['i'],b['j']]+rho[...,b['j'],b['i']]) for b in bonds],axis=-1)
    arrays={'density':density,'onsite_factorial':moment,'rho':rho,'bond_G':G,
            'meanD':moment.mean(-1),'kinetic':-2*t*G.sum(-1),'potential':.5*U*moment.sum(-1),
            'energy':np.asarray(raw['local_energy'],dtype=np.float64)}
    for axis in ('x','y'):
        indices=[k for k,b in enumerate(bonds) if b['axis']==axis]
        if indices:arrays['G'+axis]=G[...,indices].mean(-1)
    reconstructed=arrays['kinetic']+arrays['potential']-mu*density.sum(-1)
    error=float(abs(reconstructed-arrays['energy']).max())
    if error>1e-9:raise RuntimeError('stored localenergy differs from perconfiguration NN/onsite reconstruction')
    statistics={key:_stats(values,block_length) for key,values in arrays.items()}
    observed={key:stat['mean'] for key,stat in statistics.items()}
    observed.setdefault('Gx',None);observed.setdefault('Gy',None)
    observed['energy_reconstructed']=float(statistics['kinetic']['mean']+statistics['potential']['mean']-mu*np.asarray(statistics['density']['mean']).sum())
    counts={a:sum(b['axis']==a for b in bonds) for a in ('x','y')}
    observed['kinetic_from_direction_means']=-2*t*sum(counts[a]*observed['G'+a] for a in ('x','y') if counts[a])
    mean_rho=np.asarray(observed['rho'])
    return {'observed':observed,'statistics':statistics,
            'checks':{'perconfiguration_energy_reconstruction_max_abs':error,
                      'rho_raw_mean_hermiticity_max_abs':float(abs(mean_rho-mean_rho.T).max()),
                      'rho_diagonal_density_sample_max_abs':float(abs(np.diagonal(rho,axis1=-2,axis2=-1)-density).max()),
                      'kinetic_direction_reconstruction_abs':abs(observed['kinetic']-observed['kinetic_from_direction_means'])},
            'statistics_method':'Each Hermitianbond, directionmean, meanD and kinetic/potential/energy composite is formed per(time,walker) BEFORE timeblocking; no independent-error averaging. conditionalSEM=max(blockSEM,between-chainSEM), not CI/ESS/mixing proof.'}
