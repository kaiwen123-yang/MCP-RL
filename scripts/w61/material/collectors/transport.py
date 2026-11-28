"""Independent client packet memory and deterministic CPU correction at the calling collector."""
import ctypes as ct
import hashlib
import json
import mmap
import os
from pathlib import Path
import socket
import struct
import time
import numpy as np

MAX_CELLS=4200
TICKS=10
IN_BYTES=(8*(TICKS+2*TICKS*MAX_CELLS+6*MAX_CELLS)+4*TICKS*MAX_CELLS+2*MAX_CELLS+7)//8*8
OUT_BYTES=8*(12*MAX_CELLS+8*TICKS)+4+(3+TICKS)*MAX_CELLS
MESSAGE=struct.Struct('!iidi') # collector, number of cells, gamma, sequence
REPLY=struct.Struct('!iid') # action (0 local CPU, 1 shared GPU), actual batch width, synchronous service seconds
CHECKPOINTS={1,128,512,1024,1536,2048,3072,4096,6144,8192,12288,16384}


def input_views(buffer,n,ticks):
    at=0;items=[]
    for shape,typ in [((ticks,),'<f8'),((ticks,n),'<f8'),((ticks,n),'<f8'),((n,6),'<f8'),
                      ((ticks,n),'<i4'),((n,),'u1'),((n,),'u1')]:
        item=np.ndarray(shape,dtype=typ,buffer=buffer,offset=at)
        at+=item.nbytes;items.append(item)
    return items


def output_views(buffer,n,ticks):
    at=IN_BYTES;items=[]
    for shape,typ in [((n,6),'<f8'),((n,6),'<f8'),((ticks,4),'<f8'),((ticks,4),'<f8'),
                      ((1,),'<i4'),((n,),'u1'),((n,),'u1'),((n,),'u1'),((ticks,n),'u1')]:
        item=np.ndarray(shape,dtype=typ,buffer=buffer,offset=at)
        at+=item.nbytes;items.append(item)
    return items


class Client:
    def __init__(self,native,model,arm,worker,ipc,out):
        self.original=native;self.arm=arm;self.worker=worker;self.out=out
        self.stats=[];self.frames=[];self.samples={};self.seq=0;self.physical_t=0.
        self.model=model.copy()
        self.socket=None
        if arm!='cpu':
            from s1_w60_cpu_observer import ARRAY,BYTE,INDEX
            so=Path('/root/autodl-tmp/MCP-RL-w59/work/w59/m1-gpu-study/correction.so')
            self.correct=ct.CDLL(str(so)).w59_correct_last
            self.correct.argtypes=[ARRAY,ct.c_int,ct.c_int,ARRAY,ARRAY,ARRAY,INDEX,ct.c_double,ARRAY,BYTE,BYTE,ARRAY,ARRAY,BYTE,ct.c_int,ARRAY,ARRAY,ARRAY,BYTE]
            self.correct.restype=None
            fd=os.open(ipc/f'{worker}.shm',os.O_RDWR)
            self.buf=mmap.mmap(fd,IN_BYTES+OUT_BYTES);os.close(fd)
            self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET)
            self.socket.connect(str(ipc/'broker.sock'))
            self.socket.send(struct.pack('!i',worker)+hashlib.sha256(model.tobytes()).digest())
            if self.socket.recv(8)!=b'OK':raise RuntimeError('Broker material identity differs')

    def completed(self,start,end,bottom,velocity,force):
        self.original_completed(start,end,bottom,velocity,force)
        self.physical_t=float(end)
        if round(end/.0002)%10==0:
            self.frames.append((float(end),time.monotonic_ns(),float(np.sum(force))))

    def w57_packet(self,model,n,ticks,dt,normal,area,owners,gamma,state,settled,loaded,forces,stats,trace,correct,measured,diagnostics):
        self.seq+=1
        capture=self.seq in CHECKPOINTS
        if capture:
            packet=dict(n=n,ticks=ticks,dt=dt.copy(),normal=normal.copy(),area=area.copy(),owners=owners.copy(),gamma=gamma,
                     start_state=state.copy(),start_settled=settled.copy(),start_loaded=loaded.copy(),correct=correct,measured=measured.copy())
        begin=time.monotonic_ns();width=0;service=0.;dispatch=0;reply_at=begin
        if self.arm!='cpu':
            if n>MAX_CELLS or ticks!=TICKS:raise ValueError('Packet exceeds fixed actual strip address space')
            for dst,src in zip(input_views(self.buf,n,ticks),(dt,normal,area,state,owners,settled,loaded)):np.copyto(dst,src)
            self.socket.send(MESSAGE.pack(self.worker,n,gamma,self.seq))
            reply=self.socket.recv(REPLY.size)
            if len(reply)!=REPLY.size:raise RuntimeError('Collector lost its GPU batch service')
            dispatch,width,service=REPLY.unpack(reply);reply_at=time.monotonic_ns()
        if dispatch==0:
            bad=self.original.w57_packet(model,n,ticks,dt,normal,area,owners,gamma,state,settled,loaded,forces,stats,trace,correct,measured,diagnostics)
        else:
            st,before,f,s,index,sett,load,bload,tr=output_views(self.buf,n,ticks)
            bad=int(index[0])
            np.copyto(state,st);np.copyto(forces,f);np.copyto(stats,s);np.copyto(settled,sett);np.copyto(loaded,load)
            valid=ticks if bad<0 else bad//n
            if valid:np.copyto(trace[:valid],tr[:valid])
            if bad<0:
                self.correct(model,n,ticks,dt,normal,area,owners,gamma,state,settled,loaded,forces,stats,trace,correct,measured,diagnostics,before,bload)
        end=time.monotonic_ns()
        self.stats.append((self.seq,self.physical_t,n,dispatch,width,begin,reply_at,end,service))
        if capture:
            packet.update(end_state=state.copy(),end_settled=settled.copy(),end_loaded=loaded.copy(),
                          forces=forces.copy(),stats=stats.copy(),trace=trace.copy(),diagnostics=diagnostics.copy(),bad=bad)
            for k,v in packet.items():self.samples[f'p{self.seq}_{k}']=v
        return bad

    def finish(self):
        if self.socket:self.socket.close()
        np.savez_compressed(self.out/'benchmark-timing.npz',frames=np.asarray(self.frames),packets=np.asarray(self.stats),
            frame_fields='physical_end_s,monotonic_ns,measured_total_fz',packet_fields='seq,physical_s,n,dispatched_gpu,batch_width,client_start_ns,reply_ns,client_end_ns,synchronous_gpu_batch_s')
        np.savez_compressed(self.out/'parity-samples.npz',model=self.model,**self.samples)
