import numpy as np
def quatmul(a,b):
    w1,x1,y1,z1=a;w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def state_diff(ref,state):
    q=quatmul(ref[3:7]*[1,-1,-1,-1],state[3:7]);q=q/np.linalg.norm(q)
    s=np.linalg.norm(q[1:]);angle=2*np.arctan2(s,q[0]);angle=angle-2*np.pi if angle>np.pi else angle
    r=q[1:]*angle/s if s>1e-15 else 2*q[1:]
    return np.r_[state[:3]-ref[:3],r,state[7:19]-ref[7:19],state[19:]-ref[19:]]
def interp(t,ts,values):
    ts=ts[:len(values)];i=int(np.clip(np.searchsorted(ts,t,side='right')-1,0,len(ts)-1));j=min(i+1,len(ts)-1)
    a=0 if i==j else np.clip((t-ts[i])/(ts[j]-ts[i]),0,1)
    return (1-a)*values[i]+a*values[j]
def raw_policy(t,state,reference_times,reference_states,reference_actions,K,scale):
    ref=interp(t,reference_times,reference_states);ref[3:7]/=np.linalg.norm(ref[3:7]);u=interp(t,reference_times,reference_actions);kk=interp(t,reference_times,K[:-1]);dx=state_diff(ref,state)
    fb=scale*(kk@dx)
    return u,fb,dx,kk
