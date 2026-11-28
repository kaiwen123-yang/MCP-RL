"""Fuse genuine pending M1 packets from independent robot collectors into grid.y of one CUDA launch."""
import argparse,ctypes as ct,hashlib,json,mmap,os,selectors,socket,struct,sys,time
from pathlib import Path
import numpy as np
from transport import IN_BYTES,OUT_BYTES,MAX_CELLS,TICKS,MESSAGE,REPLY
W59=Path('/root/autodl-tmp/MCP-RL-w59')
sys.path.insert(0,str(W59/'scripts'))
from s1_w55_c_replay import weights,pack


def serve(ipc,spec,out):
    c=json.loads(spec.read_text())
    native=ct.CDLL(c['fused_binary']);ptr=ct.c_void_p
    native.w61_create.argtypes=[ptr,ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ptr)]
    native.w61_dispatch.argtypes=[ptr,ct.c_int,ptr,ptr,ptr,ptr]
    native.w61_destroy.argtypes=[ptr];native.w61_destroy.restype=None
    model=pack(weights(Path(c['material_weights'])))
    model_hash=hashlib.sha256(model.tobytes()).digest()
    context=ptr();err=native.w61_create(model.ctypes.data,2,MAX_CELLS,TICKS,ct.byref(context))
    if err:raise RuntimeError('CUDA context creation '+str(err))
    maps=[];addresses=[]
    for worker in range(2):
        f=(ipc/f'{worker}.shm').open('w+b');f.truncate(IN_BYTES+OUT_BYTES)
        m=mmap.mmap(f.fileno(),IN_BYTES+OUT_BYTES);f.close()
        maps.append(m);addresses.append(ct.addressof(ct.c_char.from_buffer(m)))
    poll=selectors.DefaultSelector()
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET);listener.bind(str(ipc/'broker.sock'));listener.listen(2)
    poll.register(listener,selectors.EVENT_READ,None)
    workers={};all_seen=set();pending={};records=[]
    (ipc/'ready').write_text(str(os.getpid()))
    while len(all_seen)<2 or workers:
        now=time.monotonic()
        wait=None if not pending else max(0,min(v[2] for v in pending.values())+c['coalesce_max_s']-now)
        for key,_ in poll.select(wait):
            if key.fileobj is listener:
                connection,_=listener.accept()
                hello=connection.recv(64);worker=struct.unpack('!i',hello[:4])[0]
                if worker in all_seen or hello[4:]!=model_hash:
                    connection.send(b'BAD');raise ValueError('Independent collector/material handshake differs')
                all_seen.add(worker);workers[worker]=connection
                poll.register(connection,selectors.EVENT_READ,worker);connection.send(b'OK')
            else:
                sock=key.fileobj;worker=key.data;msg=sock.recv(MESSAGE.size)
                if not msg:
                    poll.unregister(sock);sock.close();workers.pop(worker);continue
                ident,n,gamma,seq=MESSAGE.unpack(msg)
                if ident!=worker or worker in pending:raise ValueError('Non-sequential M1 packet')
                pending[worker]=(n,gamma,time.monotonic(),seq)
        if not pending:continue
        width=len(pending);start=time.monotonic()
        if width<2 and len(workers)>1 and start-min(v[2] for v in pending.values())<c['coalesce_max_s']:continue
        ids=sorted(pending);values=[pending[i] for i in ids]
        go=width>1 and sum(r[0] for r in values)>=c['minimum_joint_columns']
        sync_s=0.
        if go:
            nc=np.array([v[0] for v in values],dtype='i4');gam=np.array([v[1] for v in values],dtype='f8')
            ips=np.array([addresses[i] for i in ids],dtype=np.uintp);ops=ips+np.uintp(IN_BYTES)
            launch=time.monotonic()
            err=native.w61_dispatch(context,width,nc.ctypes.data,gam.ctypes.data,ips.ctypes.data,ops.ctypes.data)
            if err:raise RuntimeError('CUDA fused dispatch '+str(err))
            sync_s=time.monotonic()-launch
        finish=time.monotonic()
        for worker,(n,gamma,t,seq) in zip(ids,values):
            workers[worker].send(REPLY.pack(int(go),width if go else 0,sync_s))
        records.append([start,finish,int(go),width,*[pending[i][0] if i in pending else 0 for i in range(2)],
             *[pending[i][3] if i in pending else -1 for i in range(2)]])
        pending.clear()
    native.w61_destroy(context);poll.close();listener.close()
    np.savez_compressed(out/'broker.npz',dispatch=np.asarray(records),
         fields='launch_start_monotonic_s,finish_s,gpu,waiting_width,A_columns,B_columns,A_seq,B_seq',material_sha256=model_hash.hex())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('ipc',type=Path);p.add_argument('spec',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();serve(a.ipc,a.spec,a.out)
