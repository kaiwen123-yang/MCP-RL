"""Build a separate W61 CPU material binary from the immutable solver sources."""
import hashlib,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
WORK=ROOT/'work/w61/learning/cpu-exact-v1'
SRC=ROOT/'scripts'

def edited(text,before,after):
    if text.count(before)!=1: raise ValueError('Unexpected frozen solver signature: '+before[:90])
    return text.replace(before,after,1)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build():
    out=WORK/'build';out.mkdir(parents=True,exist_ok=True)
    base=(SRC/'s1_w55_c_column.cu').read_text()
    base=edited(base,'double *x,double *stress){\n    for(int i=0;i<4;++i)x[i]=prev[i];',
                    'double *x,double *stress,double *pair=nullptr){\n    for(int i=0;i<4;++i)x[i]=prev[i];')
    base=edited(base,'{*stress=s[0];return error;}',
                    '{*stress=s[0]; if(pair){pair[0]=s[0];pair[1]=s[1];} return error;}')
    (out/'s1_w55_c_column.cu').write_text(base)
    cols=(SRC/'s1_w60_columns.cc').read_text()
    cols=edited(cols,'static double mixed(', 'static double original_mixed(')
    cols=edited(cols,'double error=solve(m,prev,normal,dt,fixed,next,&first);\n    stress_pair(m,next,stress);',
                    'double error=solve(m,prev,normal,dt,fixed,next,&first,stress);\n    if (!std::isfinite(error)) stress_pair(m,next,stress);')
    cols=edited(cols,'static bool accepted(', '#include "exact_packet.hh"\n\nstatic bool accepted(')
    (out/'s1_w60_columns.cc').write_text(cols)
    packet=(SRC/'s1_w57_material_columns.cc').read_text()
    packet=edited(packet,'    const double packet_dt = std::accumulate(dt, dt+ticks, 0.);',
                         '    w61_exact::Scope scope;\n    const double packet_dt = std::accumulate(dt, dt+ticks, 0.);')
    (out/'s1_w57_material_columns.cc').write_text(packet)
    library=out/'exact-material.so'
    cmd=['g++','-O3','-std=c++17','-fPIC','-shared','-I'+str(HERE),str(out/'s1_w57_material_columns.cc'),'-o',str(library)]
    subprocess.run(cmd,check=True)
    info=dict(compiler=subprocess.check_output(['g++','-dumpfullversion'],text=True).strip(),command=cmd,
              source_files={str(p.relative_to(ROOT)):digest(p) for p in (SRC/'s1_w55_c_column.cu',SRC/'s1_w60_columns.cc',SRC/'s1_w57_material_columns.cc',HERE/'exact_packet.hh',Path(__file__))},
              binary=str(library),sha256=digest(library))
    (out/'identity.json').write_text(json.dumps(info,indent=2)+'\n')
    print(info['sha256'],flush=True)

if __name__=='__main__':build()
