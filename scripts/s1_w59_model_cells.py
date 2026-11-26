"""Causal world-fixed footprint history from observed feet and normal forces (FP64)."""
import numpy as np
from numba import njit
from s1_w59_model_fields import CELL, HALF

NX, NY = 70, 60


@njit(cache=True)
def footprint(x,y):
    keys = np.full(9,-1,np.int32)
    weights = np.zeros(9)
    if not (.8+HALF<=x<=2.2-HALF and abs(y)<=.6-HALF):
        return keys,weights
    ix0,iy0=int(np.floor((x-HALF)/CELL)),int(np.floor((y-HALF)/CELL))
    for dx in range(3):
        for dy in range(3):
            ix,iy=ix0+dx,iy0+dy
            area=max(0., min(x+HALF,(ix+1)*CELL)-max(x-HALF,ix*CELL))*max(0.,min(y+HALF,(iy+1)*CELL)-max(y-HALF,iy*CELL))
            if area>1e-15:
                k=3*dx+dy
                keys[k]=(ix-40)*NY+iy+30
                weights[k]=area/.001
    return keys,weights


@njit(cache=True)
def update_cells(table,old_keys,t,feet,fz):
    # table: last force-supported depth, last release, current owner,
    # most recent observed owner, any observed load. No plant cell label.
    keys_now=np.full((4,9),-1,np.int32)
    for leg in range(4):
        if fz[leg]<=1.:continue
        x,y,z=feet[leg]
        keys,aa=footprint(x,y)
        for a in range(9):
            key=keys[a]
            if key<0:continue
            prior=int(table[key,2])
            prior_in_this_update=False
            if prior>=0:
                for c in range(9):
                    if keys_now[prior,c]==key:prior_in_this_update=True
            if prior_in_this_update and feet[prior,2]<=z:continue
            keys_now[leg,a]=key
            table[key,0]=max(.022-z,0.)
            table[key,2]=leg
            table[key,3]=leg
            table[key,4]=1.
    for leg in range(4):
        for j in range(9):
            key=old_keys[leg,j]
            if key<0:continue
            if not (keys_now==key).any():
                table[key,1]=t
                table[key,2]=-1
    return keys_now


@njit(cache=True)
def query_cells(table,t,xy):
    features=np.zeros((len(xy),4)); valid=np.zeros(len(xy))
    for i in range(len(xy)):
        keys,weights=footprint(xy[i,0],xy[i,1])
        for n in range(9):
            k=keys[n]
            if k<0:continue
            w=weights[n]
            valid[i]+=w
            if table[k,2]>=0:
                features[i,0]+=w
            elif table[k,4]>0 and table[k,1]>=0:
                age=max(0.,t-table[k,1])
                features[i,1]+=w*table[k,0]
                features[i,2]+=w*age
                features[i,3]+=w
        if features[i,3]>0:
            features[i,2]/=features[i,3]
    return features,valid


@njit(cache=True)
def series(feet,fz,target,target_valid,step):
    n=(len(feet)-1)//step+1
    arrays=np.zeros((n,8,4));valid=np.zeros((n,8))
    table=np.zeros((NX*NY,5));table[:,1:4]=-1
    active=np.full((4,9),-1,np.int32)
    for i in range(len(feet)):
        t=i*.002
        active=update_cells(table,active,t,feet[i],fz[i])
        if i%step==0:
            j=i//step
            points=np.vstack((feet[i,:,:2].copy(),target[j]))
            arr,mask=query_cells(table,t,points)
            for leg in range(4):
                if not target_valid[j,leg]:
                    arr[leg+4,:]=0.
                    mask[leg+4]=0.
            arrays[j]=arr
            valid[j]=mask
    return arrays,valid


class CausalCellMap:
    def __init__(self):
        self.table=np.zeros((NX*NY,5),dtype=np.float64)
        self.table[:,1:4]=-1
        self.active=np.full((4,9),-1,np.int32)

    def observe(self,t,feet,measured_force):
        self.active=update_cells(self.table,self.active,float(t),np.asarray(feet,dtype=np.float64).reshape(4,3),np.asarray(measured_force).reshape(4,3)[:,2])

    def query(self,t,feet,targets,target_valid):
        xy=np.vstack((np.asarray(feet).reshape(4,3)[:,:2],np.asarray(targets).reshape(4,2)))
        mem,mask=query_cells(self.table,t,xy)
        mem[4:]*=np.asarray(target_valid).reshape(4,1)
        mask[4:]*=np.asarray(target_valid)
        return mem,mask
