"""Analytic capped-drive-free local diagnostic at archived physical joint mobility spectra."""
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[3]
FROZEN=ROOT.parent/'MCP-RL-w59'
REL=Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
ROWS=FROZEN/REL/'w59/model/historical-origins.json'


def main():
    h=.02;k=60.;d=5.
    result=[]
    for source in json.loads(ROWS.read_text())['episodes']:
        ep=source['episode']
        inp=FROZEN/'work/w59/model/historical-origins'/(ep+'.npz')
        with np.load(inp) as dat:rigid=dat['rigid'];ids=dat['index']
        for operator,ident in zip(rigid,ids):
            B=operator[6:,1:13].astype(float)
            # Rotor/geometric constraints and physical clipping are held out only
            # for this original linear, interior, local PD subproblem.
            for mobility in np.linalg.eigvalsh(.5*(B+B.T)):
                a,b=mobility*k,mobility*d
                continuous=np.array([[0.,1.],[-a,-b]])
                exact=expm(h*continuous)
                explicit=np.array([[1-.5*h*h*a,h-.5*h*h*b],[-h*a,1.-h*b]])
                den=1+h*b+.5*h*h*a
                vq=-h*a/den;vv=(1.-.5*h*h*a)/den
                implicit=np.array([[1+.5*h*vq,.5*h*(1.+vv)],[vq,vv]])
                metric=np.diag([np.sqrt(k),1./np.sqrt(mobility)])
                inv=np.diag(1./np.diag(metric))
                norm=np.linalg.norm(metric@exact@inv)
                row=[mobility]
                for A in (explicit,implicit):
                    row.extend([float(max(abs(np.linalg.eigvals(A)))),np.linalg.norm(metric@(A-exact)@inv)/norm])
                result.append((ep,int(ident),row))
    arr=np.array([v[-1] for v in result])
    out=ROOT/REL/'w61/material/w61-a2-pd-modal-v1'
    out.mkdir(parents=True,exist_ok=True)
    columns=['mobility','explicit_radius','explicit_energy_metric_local_error','implicit_radius','implicit_energy_metric_local_error']
    np.savez_compressed(out/'spectra.npz',values=arr,episode=[x[0] for x in result],origin=[x[1] for x in result],columns=columns)
    summary=dict(runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        serial='Frozen archived M^-1[S^T] actuated block, each selected physical origin, linear 60/5 drive plus held target perturbation with no saturation on its interior. Inertia eigenmodes within an origin are correlated.',
        source=str(ROWS),h=h,K=k,D=d,modes=len(arr),distinct_origins=len({(a,i) for a,i,_ in result}),
        explicit_radius_above_one=int((arr[:,1]>1).sum()),implicit_radius_above_one=int((arr[:,3]>1).sum()),
        p50_p95_max={n:np.percentile(arr[:,i],[50,95,100]).tolist() for i,n in enumerate(columns)},
        actual_robot_steps=0,limits='The full plant includes moving configurations, unilateral contacts and clamps; this diagnostic is local to each interior PD mode.')
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary['explicit_radius_above_one'],summary['implicit_radius_above_one'],len(arr),flush=True)


if __name__=='__main__':main()
