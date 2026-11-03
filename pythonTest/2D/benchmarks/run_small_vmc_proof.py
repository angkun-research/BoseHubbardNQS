"""The predeclared two-model2x3 stochastic ridgeSR+nativeSGD proof.

Existing train/evaluate CLIs only. Fresh seed123, no warmstart, noED labels or
EDreference flag in training. Exact426state diagnostics run outside optimization.
Only initial/final weights exist: the saved training curve is MonteCarlo, not an
exact-energy trajectory. FrozenCPUfloat64 seeds321/654 are both retained.
"""
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
import subprocess,sys,time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import build_model,save_model,load_model,parameter_count
from lattice import bonds_2d
from exact_diagonalization import count_basis_states,exact_ground_state
from leakage_analysis import evaluate_logs
from vmc import check_sr_budget

R=ROOT/'benchmarks/results/2026-11-01-physics-focus'
O=ROOT/'outputs/2026-11-01-physics-focus'
STATUS=R/'small-vmc-launch-status.json'
RESULTS=R/'small-vmc-proof-results.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


@torch.no_grad()
def exact_metrics(model,raw,basis,H,reference_energy):
    model=model.double().eval()
    f=np.concatenate([evaluate_logs(model,basis[start:start+256]) for start in range(0,len(basis),256)])
    psi=np.exp(f-f.max());psi/=np.linalg.norm(psi)
    p=psi**2;applied=H@psi;energy=float(psi@applied)
    density=p@basis;moment=p@(basis*(basis-1))
    # Independent row-action ladder sum; no sampling estimator or division by
    # small model amplitudes. rho[i,j]=<b_i^dag b_j>, basis i=y*Lx+x.
    lookup={tuple(n):k for k,n in enumerate(basis)}
    rho=np.zeros((6,6),dtype=np.float64)
    for i in range(6):
        rho[i,i]=density[i]
        for j in range(6):
            if i==j:continue
            for k,n in enumerate(basis):
                if n[i]==0 or n[j]==4:continue
                other=n.copy();other[i]-=1;other[j]+=1
                rho[i,j]+=psi[k]*psi[lookup[tuple(other)]]*math.sqrt(float(n[i]*(n[j]+1)))
    reconstructed=-sum(rho[i,j]+rho[j,i] for i,j in bonds_2d(2,3))+16.74/2*moment.sum()
    assert abs(reconstructed-energy)<1e-10
    return {'energy':energy,'energy_per_site':energy/6,'variance':float(np.linalg.norm(applied-energy*psi)**2),
            'F_N0':float(abs(raw@psi)),'absolute_overlap_convention':'unsquared amplitude overlap with raw normalized canonical ED',
            'energy_error_to_ed':energy-reference_energy,'energy_error_per_site':(energy-reference_energy)/6,
            'density':density.tolist(),'onsite_factorial_moment':moment.tolist(),'rho':rho.tolist(),
            'energy_from_observables':float(reconstructed),'rho_raw_hermiticity_max_abs':float(abs(rho-rho.T).max()),
            'rho_trace':float(np.trace(rho)),'norm':float(np.linalg.norm(psi)),
            'expectation':'exact full426state model probability weighting, CPUfloat64; not MC',
            'forward_dtype':'torch.float64'}


def _conditional_sem(stats):
    candidates=[np.asarray(stats[k],dtype=np.float64) for k in ('block_standard_error','between_chain_standard_error') if stats.get(k) is not None]
    return np.maximum.reduce(candidates).tolist() if candidates else None


def main():
    if STATUS.exists() or RESULTS.exists():raise RuntimeError('refuse existing smallVMC run/results')
    R.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/bhnqs-2d-mpl')
    inventory=R/'checker-evidence-inventory.json'
    assert inventory.is_file()
    assert count_basis_states(6,6,4)==426 and len(bonds_2d(2,3))==9
    status={'status':'prepared_before_launch','recorded_before_launch':now(),'expected_run_count':2,'completed_run_count':0,'failed_run_count':0,
            'checker_inventory':str(inventory),'checker_inventory_sha256':sha(inventory),
            'protocol':str(R/'small-vmc-conditional-protocol.json'),'protocol_sha256':sha(R/'small-vmc-conditional-protocol.json'),
            'source_sha256_at_start':{name:sha(ROOT/name) for name in ('train.py','vmc.py','models.py','evaluate.py','observables.py','lattice.py','exact_diagonalization.py','benchmarks/run_small_vmc_proof.py')},
            'resources':{'training_jobs_max':2,'CPU_threads_per_job':1,'device':'cpu','model_dtype':'float64','SR_dtype':'float64','GPU_jobs':0},
            'scope':'Only two predeclared fresh-init small-system proof runs; no optimizationgrid, supervisedwarmstart, EDlabels or large-sizecampaign.',
            'runs':[]}
    result={'status':'prepared','recorded_at':now(),'geometry':{'Lx':2,'Ly':3,'pbc_x':True,'pbc_y':True,'Ntot':6,'Nmax':4,'t':1.,'U':16.74,'mu':0.,'unique_bonds':9,'dimension':426},
            'protocol':str(R/'small-vmc-conditional-protocol.json'),'source_sha256_at_start':status['source_sha256_at_start'],
            'exact_trajectory_scope':'Initial/final exact points only; train CLI does not retain intermediateweights. trajectory_MC is stochastic pre-update energy/variance, not an exactcurve.',
            'interpretation':'ED is diagnostic only and never enters gradient/update calls. Execution completion does not itself establish convergence; failures/missedaccuracy retained without retuning.',
            'models':[]}
    def persist():
        for path,value in ((STATUS,status),(RESULTS,result)):
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    # Concrete commands and hashes exist before eigensolve or optimizer subprocess.
    for arch in ('FCNet','Conv2DNet'):
        short='FC32' if arch=='FCNet' else 'Conv32'
        checkpoint=O/f'{short}-2x3-srsgd500-seed123.pt';initial=O/f'{short}-2x3-initial-seed123.pt'
        log=R/f'{short}-2x3-srsgd500-seed123.log'
        if any(p.exists() for p in (checkpoint,Path(str(checkpoint)+'.json'),initial,log)):
            raise RuntimeError('refuse existing checkpoint/log')
        command=[sys.executable,'-u',str(ROOT/'train.py'),'--arch',arch,'--hidden-dim','32',
         '--Lx','2','--Ly','3','--Ntot','6','--Nmax','4','--pbc-x','--pbc-y','--t','1','--U','16.74','--mu','0',
         '--update-method','sr','--optimizer','sgd','--sr-regularizer','ridge','--epochs','500','--walkers','64','--steps','16',
         '--initial-burnin','4096','--burnin','64','--thin','8','--lr','.02','--lr-end','.002','--tau','.001','--tau-end','.0001',
         '--jacobian-method','vmap','--jacobian-batch-size','128','--device','cpu','--dtype','float64','--seed','123','--output',str(checkpoint)]
        assert '--warm-start' not in command and '--ed-reference' not in command
        row={'label':short,'family':arch,'command':command,'checkpoint':str(checkpoint),'initial_checkpoint':str(initial),'log':str(log),'status':'prepared'}
        status['runs'].append(row)
        result['models'].append({'family':arch,'label':short,'checkpoint':str(checkpoint),'initial_checkpoint':str(initial),
            'exact_initial':None,'exact_final':None,'trajectory_MC':[],'frozen':{},'status':'prepared'})
    persist()
    active=[]
    try:
        # This reference is outside the optimizer child and never passed as data.
        E0,raw,basis,H=exact_ground_state(2,3,6,4,1.,16.74,0.,True)
        raw=np.asarray(raw,dtype=np.float64);raw/=np.linalg.norm(raw);basis=np.asarray(basis,dtype=np.int64)
        result['ED_reference']={'energy':float(E0),'dimension':426,'residual_norm':float(np.linalg.norm(H@raw-E0*raw)),
                                'scope':'Diagnostics only. Training child receives no ED data/path/labels.'}
        for row,modelrow in zip(status['runs'],result['models']):
            # Exactly matches train.main constructor: manual_seed, defaultFP32
            # model construction then .to(float64), with no prior random draws.
            torch.manual_seed(123)
            model=build_model(row['family'],2,3,Nmax=4,hidden_dim=32,pbc_x=True,pbc_y=True).double().eval()
            save_model(row['initial_checkpoint'],model,6,1.,16.74,0.)
            row['initial_checkpoint_sha256']=sha(row['initial_checkpoint'])
            row['parameter_count']=parameter_count(model)
            row['SR_budget']=check_sr_budget(model,1024,jacobian_method='vmap',jacobian_batch_size=128)
            modelrow['parameter_count']=parameter_count(model)
            modelrow['exact_initial']=exact_metrics(model,raw,basis,H,float(E0))
            modelrow['initial_checkpoint_sha256']=row['initial_checkpoint_sha256']
            modelrow['initialization_provenance']='Replicated exact freshCLIconstructor seed123, CPUdefaultFP32init then castfloat64; checkpoint for diagnostics ONLY, not loaded by training.'
            persist()
        # Exactly two CPUthreads1 optimizer subprocesses, no additional trials.
        for row,modelrow in zip(status['runs'],result['models']):
            log=open(row['log'],'x');started=time.perf_counter()
            process=subprocess.Popen(row['command'],cwd=ROOT.parents[1],env=env,stdout=log,stderr=subprocess.STDOUT)
            row.update(status='running',pid=process.pid,started_at=now());modelrow['status']='training'
            active.append((row,modelrow,process,log,started));status['status']=result['status']='training';persist()
            print('launched',row['label'],process.pid,flush=True)
        while active:
            for item in list(active):
                row,modelrow,process,log,started=item;code=process.poll();row['subprocess_wall_seconds']=time.perf_counter()-started
                if code is None:continue
                log.close();active.remove(item);row.update(status='completed' if code==0 else 'execution_failed',exit_code=code,finished_at=now())
                if code:
                    modelrow['status']='training_execution_failed';status['failed_run_count']+=1
                else:
                    status['completed_run_count']+=1
                    report_path=Path(row['checkpoint']+'.json');report=json.loads(report_path.read_text())
                    retained=R/f"{row['label']}-2x3-srsgd500-trainer.json"
                    with retained.open('x') as stream:stream.write(report_path.read_text())
                    model,_=load_model(Path(row['checkpoint']),'cpu');model=model.double()
                    modelrow.update(status='exact_final_ready',checkpoint_sha256=sha(row['checkpoint']),
                        exact_final=exact_metrics(model,raw,basis,H,float(E0)),trajectory_MC=report['history'],
                        trainer_report=str(retained),training_wall_seconds=row['subprocess_wall_seconds'],
                        CLI_training_and_final_seconds=report['training_and_final_measurement_seconds'])
                    assert len(report['history'])==500
                    assert report['settings']['optimizer']=='sgd' and report['settings']['update_method']=='sr' and report['sr_regularizer']=='ridge'
                    print('trained',row['label'],'exactE',modelrow['exact_final']['energy'],'F',modelrow['exact_final']['F_N0'],flush=True)
                persist()
            if active:time.sleep(2)
        # Sequential frozen tasks keep totalCPUjobs low; both seeds always run.
        status['status']=result['status']='frozen_evaluations';persist()
        for row,modelrow in zip(status['runs'],result['models']):
            if row['status']!='completed':continue
            for seed in (321,654):
                output=O/f"frozen/{row['label']}-2x3-srsgd500-seed{seed}.json"
                logpath=R/f"{row['label']}-2x3-frozen-seed{seed}.log"
                command=[sys.executable,'-u',str(ROOT/'evaluate.py'),'--checkpoint',row['checkpoint'],'--device','cpu','--dtype','float64',
                         '--walkers','64','--steps','512','--burnin','4096','--thin','8','--block-length','64','--full-rho',
                         '--batch-size','256','--seed',str(seed),'--output',str(output)]
                info={'status':'prepared','command':command,'report':str(output),'log':str(logpath)}
                modelrow['frozen'][str(seed)]=info;persist()
                started=time.perf_counter()
                with logpath.open('x') as log:
                    process=subprocess.Popen(command,cwd=ROOT.parents[1],env=env,stdout=log,stderr=subprocess.STDOUT)
                    info.update(status='running',pid=process.pid);persist();code=process.wait()
                info.update(exit_code=code,wall_seconds=time.perf_counter()-started)
                if code:
                    info['status']='execution_failed';modelrow['status']='frozen_execution_failed';persist();continue
                frozen=json.loads(output.read_text());assert frozen['checkpoint_sha256']==modelrow['checkpoint_sha256']
                retained=R/f"{row['label']}-2x3-frozen-seed{seed}.json"
                with retained.open('x') as stream:stream.write(output.read_text())
                exact=modelrow['exact_final'];density=np.asarray(frozen['density_all_samples']);rho=np.asarray(frozen['one_body_matrix_all_samples'])
                E_MC=frozen['statistics']['energy']['mean']
                info.update(status='completed',retained_report=str(retained),statistics=frozen['statistics'],sampling=frozen['sampling'],
                    comparison_to_exact={'MC_energy_mean':E_MC,'exact_model_energy':exact['energy'],'difference':E_MC-exact['energy'],
                     'conditional_energy_SEM':_conditional_sem(frozen['statistics']['energy']),
                     'density_MC_mean':density.tolist(),'density_exact':exact['density'],'density_difference':(density-exact['density']).tolist(),
                     'density_max_abs_difference':float(abs(density-exact['density']).max()),
                     'rho_MC_mean':rho.tolist(),'rho_exact':exact['rho'],'rho_difference':(rho-exact['rho']).tolist(),
                     'rho_max_abs_difference':float(abs(rho-exact['rho']).max()),
                     'density_conditional_SEM':_conditional_sem(frozen['statistics']['density']),
                     'rho_pair_order_conditional_SEM':_conditional_sem(frozen['statistics']['one_body']),
                     'uncertainty':'Conditional block/chain SEM, no ESS/CI/stationarity claim; comparison is to exact expectation of SAMEfinalmodel, not only EDgroundstate.'})
                print('frozen',row['label'],seed,'E_MC-exact',E_MC-exact['energy'],flush=True);persist()
            if all(q['status']=='completed' for q in modelrow['frozen'].values()):modelrow['status']='completed'
        for name,digest in status['source_sha256_at_start'].items():assert sha(ROOT/name)==digest,name
        result['status']='completed' if all(q['status']=='completed' for q in result['models']) else 'completed_with_failures_retained'
        status['status']=result['status'];status['finished_at']=now();persist()
    except BaseException as error:
        status.update(status='runner_failed',error=f'{type(error).__name__}: {error}');result.update(status='failed_retained',error=status['error']);persist();raise
    print(RESULTS,flush=True)
    return result


if __name__=='__main__':main()
