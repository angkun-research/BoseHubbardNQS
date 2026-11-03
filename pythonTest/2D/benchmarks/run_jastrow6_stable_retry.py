"""One separately approved conservative retry; original failed evidence is kept."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'benchmarks/results/2026-11-01-backup-extension'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=json.loads((R/'Jastrow-basic-6x4-stable-retry-protocol.json').read_text())
    approval=R/'checker-jastrow6-retry-readiness.json'
    if not json.loads(approval.read_text()).get('approved',False):
        raise RuntimeError('retry checker approval pending')
    for name,digest in protocol['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise RuntimeError('source changed since retry predeclaration')
    output=R/'Jastrow-basic-6x4-stable-retry-results.json'
    if output.exists() or Path(protocol['checkpoint']).exists():
        raise RuntimeError('refuse existing retry')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/bhnqs-2d-mpl')
    result={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
            'protocol':str(R/'Jastrow-basic-6x4-stable-retry-protocol.json'),
            'failed_predecessor':protocol['failed_predecessor'],'source_sha256_at_start':protocol['source_sha256'],
            'checkpoint':protocol['checkpoint'],'parameter_count':430,'runs':{},'frozen':{}}
    def persist():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    def execute(label,command):
        log=R/(label+'.log');started=time.perf_counter()
        with log.open('x') as stream:
            child=subprocess.Popen(command,env=env,cwd=ROOT.parents[1],stdout=stream,stderr=subprocess.STDOUT)
            row={'status':'running','command':command,'log':str(log),'pid':child.pid}
            result['runs'][label]=row;persist();code=child.wait()
        row.update(status='completed' if code==0 else 'execution_failed',exit_code=code,wall_seconds=time.perf_counter()-started)
        persist()
        if code:
            raise RuntimeError(f'{label} execution failed; retain log')
    persist()
    try:
        execute(protocol['label'],protocol['command'])
        cp=Path(protocol['checkpoint']);result['checkpoint_sha256']=sha(cp)
        retained=R/(protocol['label']+'-trainer.json')
        with retained.open('x') as stream:stream.write(Path(str(cp)+'.json').read_text())
        result['trainer_report']=str(retained);persist()
        # Fixed final checkpoint; both seeds, same frozen protocol as failed arm.
        for seed,command in protocol['frozen_commands'].items():
            execute(protocol['label']+'-frozen'+seed,command)
            report=Path(command[command.index('--output')+1])
            data=json.loads(report.read_text())
            if data['checkpoint_sha256']!=result['checkpoint_sha256']:
                raise RuntimeError('frozen measurement loaded different weights')
            retained=R/(protocol['label']+'-frozen'+seed+'.json')
            if retained.exists():raise RuntimeError('refuse existing retained report')
            shutil.copyfile(report,retained)
            result['frozen'][seed]={'report':str(retained),'raw_samples':str(report)+'.samples.npz'};persist()
        execute(protocol['label']+'-assessment',protocol['assessment_command'])
        result['assessment']=protocol['assessment_command'][-1]
        result['assessment_summary']=json.loads(Path(result['assessment']).read_text())
        result['status']='completed'
    except BaseException as error:
        result.update(status='execution_failed_retained',error=f'{type(error).__name__}: {error}')
        persist();raise
    result['finished_at']=datetime.now(timezone.utc).isoformat();persist()


if __name__=='__main__':
    main()
