"""r60 FP64 CPU observation on the ten completed, world-fixed substeps."""
import ctypes as ct
import faulthandler
import json
import mmap
import os
from pathlib import Path
import signal
import struct
import time

import numpy as np

from s1_w55_c_replay import pack, weights
from s1_w56_window import WindowObserver
from tm02.reference import IntegrationFailure


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT/'work/w56-r60/columns.so'
ARRAY = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
BYTE = np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS')
INDEX = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
OFFSETS = np.array([(dx, dy) for dx in range(3) for dy in range(3)])


class CallMarker:
    # The mmap is overwritten in place. SIGUSR1 requests a read-only all-thread
    # traceback only after registration here. No watchdog, timeout or trigger.
    layout = struct.Struct('<Qdd40sQ')
    def __init__(self, directory):
        self.file = (directory/'r60-current-call.bin').open('w+b')
        self.file.truncate(self.layout.size)
        self.page = mmap.mmap(self.file.fileno(), self.layout.size)
        self.generation = 0
        self.stacks = (directory/'r60-python-stacks.log').open('a')
        faulthandler.register(signal.SIGUSR1, file=self.stacks, all_threads=True)
        self.mark('r60_attached', 0.)
        (directory/'r60-call-format.json').write_text(json.dumps(dict(pid=os.getpid(),
            signal='SIGUSR1', format=self.layout.format, fields=['sequence','physical_s','unix_s','call','sequence_copy'],
            clock='double seconds', compare_sequences='Read again if sequence and sequence_copy differ',
            stack='r60-python-stacks.log'), indent=2)+'\n')

    def mark(self, label, t):
        self.generation += 1
        self.layout.pack_into(self.page, 0, self.generation, t, time.time(), label.encode(), self.generation)


class CPUColumns:
    first_call_including_compile_s = None  # compiled offline by build_s1_w60_columns.sh
    def __init__(self, path):
        self.model = pack(weights(path))
        self.lib = ct.CDLL(str(LIBRARY))
        self.lib.w60_step.argtypes = [ARRAY,ct.c_int,ARRAY,ARRAY,ct.c_double,BYTE,ARRAY]
        self.lib.w60_packet.argtypes = [ARRAY,ct.c_int,ct.c_int,ARRAY,ARRAY,ARRAY,INDEX,
                                        ct.c_double,ARRAY,BYTE,BYTE,ARRAY,ARRAY]
        self.lib.w60_recovery.argtypes = [ARRAY,ct.c_int,ARRAY,ARRAY,ARRAY]

    def step(self, e, q, normal, dt, fixed):
        out = np.empty((len(e),7))
        i = self.lib.w60_step(self.model,len(e),np.column_stack((e,q)),
                              np.ascontiguousarray(normal,dtype=float),dt,
                              np.ascontiguousarray(fixed,dtype='u1'),out)
        if i >= 0:
            raise IntegrationFailure(i,out[i,6],30)
        return out[:,:2],out[:,2:4],out[:,4:6],out[:,6]

    def recovery(self, e, q):
        out = np.empty((len(e),3)); failure=np.zeros(1)
        i=self.lib.w60_recovery(self.model,len(e),np.column_stack((e,q)),out,failure)
        if i>=0:
            raise IntegrationFailure(i,failure[0],30)
        return out


def grow_cells(g, keys):
    """Append new material world cells once, in the original encounter order."""
    if not keys:
        return
    start=len(g.e); n=len(keys)
    g.ids.update((k,start+j) for j,k in enumerate(keys))
    g.xy.extend(((ix+.5)*g.cell,(iy+.5)*g.cell) for ix,iy in keys)
    for field in ('e','q','stress'):
        setattr(g,field,np.concatenate((getattr(g,field),np.zeros((n,2)))))
    for field,value in (('loaded',False),('last_owner',-1),('last_release',np.nan),
                        ('ever_loaded',False),('last_area',0.),('last_normal',0.),('last_load_event_id',-1)):
        a=getattr(g,field)
        setattr(g,field,np.concatenate((a,np.full(n,value,dtype=a.dtype))))


class CPUWindowObserver(WindowObserver):
    def __init__(self, parameters, model_path, law=None):
        super().__init__(parameters,None,law=law if law is not None else CPUColumns(model_path))
        self.relaxing=set()
        self.marker=None
        self.last_forces=np.zeros((10,4)); self.last_areas=np.zeros((10,4))
        self.integration_column_steps=0
        self.max_packet_columns=0

    def mark(self, label, t):
        if self.marker is not None:self.marker.mark(label,t)

    def geometry(self, bottom):
        g=self.ground
        xy=bottom[:,:,:2]
        interior=((xy[:,:,0]>=.8+g.half)&(xy[:,:,0]<=2.2-g.half)&(np.abs(xy[:,:,1])<=.6-g.half))
        if not np.any(interior):
            return np.empty(0,int),np.empty(0,int),np.empty(0,int),np.empty(0)
        lo=xy-g.half;hi=xy+g.half
        ij=np.floor(lo/g.cell).astype(int)[:,:,None,:]+OFFSETS
        overlap=np.maximum(np.minimum(hi[:,:,None,:],(ij+1)*g.cell)-np.maximum(lo[:,:,None,:],ij*g.cell),0.)
        sizes=overlap[:,:,:,0]*overlap[:,:,:,1]
        k,foot,cell=np.nonzero(interior[:,:,None] & (sizes>0))
        # Unique integer code on this bounded strip, including negative Y.
        codes=ij[k,foot,cell,0]*1000+ij[k,foot,cell,1]+200
        unique,first,inverse=np.unique(codes,return_index=True,return_inverse=True)
        keys=[(int(code//1000),int(code%1000-200)) for code in unique]
        grow_cells(g,[keys[j] for j in np.argsort(first) if keys[j] not in g.ids])
        ids=np.array([g.ids[key] for key in keys],int)[inverse]
        area=sizes[k,foot,cell]
        # Sort by time and cell then lowest z, preserving leg number on ties.
        order=np.lexsort((foot,bottom[k,foot,2],ids,k))
        k,foot,ids,area=k[order],foot[order],ids[order],area[order]
        chosen=np.r_[True,(k[1:]!=k[:-1])|(ids[1:]!=ids[:-1])]
        return k[chosen],foot[chosen],ids[chosen],area[chosen]

    def observe(self,t,bottom,velocity):
        if len(self.packet)<10:return
        if len(self.packet)!=10 or abs(self.packet[-1][1]-t)>1e-9:
            raise ValueError(f'Missed a complete 500Hz window at {t}')
        self.mark('material_packet',t)
        begin=time.perf_counter(); g=self.ground
        self.packet_start=self.packet[0][0];self.packet_last=self.packet[-1][0]
        ends=np.array([record[1] for record in self.packet])
        dt=np.diff(np.r_[self.time,ends])
        feet=np.stack([record[2] for record in self.packet])
        tick,leg,global_ids,area=self.geometry(feet)
        ids=np.array(sorted(self.relaxing.union(global_ids)),int)
        n=len(ids)
        forces=np.zeros((10,4)); self.last_areas.fill(0.)
        if n:
            subset=np.searchsorted(ids,global_ids)
            normal=np.full((10,n),np.inf);areas=np.zeros((10,n));owner=np.full((10,n),-1,np.int32)
            normal[tick,subset]=feet[tick,leg,2]/g.h
            areas[tick,subset]=area;owner[tick,subset]=leg
            np.add.at(self.last_areas,(tick,leg),area)
            state=np.column_stack((g.e[ids],g.q[ids],g.stress[ids]))
            settled=np.array([i not in self.relaxing for i in ids],dtype='u1')
            loaded=np.zeros(n,dtype='u1');stats=np.zeros((10,4))
            bad=g.law.lib.w60_packet(g.law.model,n,10,dt,normal,areas,owner,g.gamma,state,settled,loaded,forces,stats)
            if bad>=0:
                raise IntegrationFailure(bad//n,stats[0,3],30)
            g.e[ids]=state[:,:2];g.q[ids]=state[:,2:4];g.stress[ids]=state[:,4:6];g.loaded[ids]=loaded
            self.relaxing=set(ids[settled==0])
            peak=stats.max(axis=0)
            g.max_strain=max(g.max_strain,peak[0]);g.max_stress=max(g.max_stress,peak[1])
            g.max_rate=max(g.max_rate,peak[2]);g.max_residual=max(g.max_residual,peak[3])
            exited=np.flatnonzero(stats[:,0]>.02)
            if len(exited) and g.domain_exit is None:
                j=exited[0];g.domain_exit=dict(time_s=ends[j],max_abs_principal_strain=stats[j,0])
        self.last_forces=forces
        self.integration_column_steps+=10*n;self.max_packet_columns=max(self.max_packet_columns,n)
        self.windows+=1;self.updates+=10;self.time=float(ends[-1]);self.packet.clear()
        self.wall_s+=time.perf_counter()-begin;self.mark('packet_published',t)

    def recovery(self, query_xy):
        self.mark('material_query',self.time)
        begin=time.perf_counter();g=self.ground
        # Footprint lookup with create=False omits virgin cells, which contribute
        # exactly zero to the numerator. Keep the original whole-sole denominator.
        maps=[g.footprint(x,create=False) for x in query_xy]
        ids=np.array(sorted({i for ii,_ in maps for i in ii}),int)
        heights=g.law.recovery(g.e[ids],g.q[ids])
        values=np.zeros((4,3))
        for i,(ii,aa) in enumerate(maps):
            if ii:values[i]=(-g.h/g.area)*np.dot(aa,heights[np.searchsorted(ids,ii)])
        self.query_calls+=1;self.query_wall_s+=time.perf_counter()-begin
        self.mark('query_returned',self.time)
        return values,np.array([g.interior(x) for x in query_xy])

    def timing(self):
        return super().timing() | dict(backend='CPU FP64 frozen potentials; central finite-difference Newton',
                  sparse_global_column_steps=self.integration_column_steps,max_packet_columns=self.max_packet_columns)
