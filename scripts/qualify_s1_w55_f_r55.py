"""Qualify the r55 A controller on its OWN complete original E30 recording."""
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from publish_s1_go2_w54 import write
from run_s1_w55_f import SMALL, LARGE

TRAJECTORY_FIELDS = ('time','qpos','qvel','q_target','contact','actual_grf','total_ground_force',
                     'nonfoot_ground_force','feet','feet_velocity','phase','command_velocity','command_omega')
PHYSICAL_FIELDS = ('physical',)


def compare_array(x, y):
    if x.shape != y.shape:
        return dict(equal=False,source_shape=list(x.shape),new_shape=list(y.shape))
    same=bool(np.array_equal(x,y))
    return dict(equal=same,shape=list(y.shape),different_values=int(np.count_nonzero(x!=y)))


def paired_video(previous, candidate):
    reference=imageio.get_reader(previous)
    new=imageio.get_reader(candidate)
    widths=[]
    unequal_frames=[]
    first_difference=None
    frame_count=0
    try:
        from itertools import zip_longest
        for i,(a,b) in enumerate(zip_longest(reference,new)):
            frame_count=i+1
            if a is None or b is None or a.shape != b.shape:
                if first_difference is None:
                    first_difference=dict(frame=i,reference_shape=None if a is None else list(a.shape),
                                          new_shape=None if b is None else list(b.shape))
                unequal_frames.append(i)
                continue
            if not widths:
                widths=list(b.shape)
            # Renderer draws name, stage and clock in the first text line (y=9).
            # All pixels in rows 32..719 include the robot, terrain and rest of HUD.
            if not np.array_equal(a[32:],b[32:]):
                unequal_frames.append(i)
                if first_difference is None:
                    delta=np.any(a[32:]!=b[32:],axis=2)
                    yy,xx=np.where(delta)
                    first_difference=dict(frame=i,changed_pixels=int(delta.sum()),
                                          bounding_box_xyxy=[int(xx.min()),int(yy.min()+32),int(xx.max()),int(yy.max()+32)])
    finally:
        reference.close();new.close()
    return dict(equal=not unequal_frames,decoded_frame_pairs=frame_count,image_shape=widths,
                compared_rows='32 through the final decoded row; full robot scene and lower HUD',
                differing_frames=len(unequal_frames),first_difference=first_difference)


def review():
    reviewed=json.loads((SMALL/'r53-selection.json').read_text())
    old=reviewed['attempt']
    stage=json.loads((SMALL/'F-A-r55/E30.json').read_text())
    new=stage['attempt']
    previous=json.loads((SMALL/(old+'.json')).read_text())
    actual=json.loads((SMALL/(new+'.json')).read_text())
    motion=json.loads((SMALL/(new+'-analysis.json')).read_text())
    if reviewed['normal_sustained_motion'] is not True or stage['status']!='passed' or not stage.get('video'):
        raise ValueError('An independently passed and fully rendered r55 A E30 is required')
    aroot=LARGE/old;broot=LARGE/new
    arrays={}
    with np.load(aroot/'trajectory.npz') as a,np.load(broot/'trajectory.npz') as b:
        for key in TRAJECTORY_FIELDS:
            arrays[key]=compare_array(a[key],b[key])
    names_old=sorted(p.name for p in aroot.glob('physical-*.npz'))
    names_new=sorted(p.name for p in broot.glob('physical-*.npz'))
    physical={}
    for name in sorted(set(names_old)|set(names_new)):
        if not (aroot/name).is_file() or not (broot/name).is_file():
            physical[name]=dict(equal=False)
            continue
        with np.load(aroot/name) as a,np.load(broot/name) as b:
            physical[name]=compare_array(a['physical'],b['physical'])
    av=json.loads((aroot/'video-asset.json').read_text());bv=json.loads((broot/'video-asset.json').read_text())
    video=paired_video(Path(av['path']),Path(bv['path']))
    new_conditions={k:actual[k] for k in ('success','stage_status','evaluation_stage','stage_progress','legal_route','physical_samples','simulation_s')}
    conditions=all(v==previous[k] for k,v in new_conditions.items())
    normal=bool(conditions and all(a['equal'] for a in arrays.values()) and
                physical and all(v['equal'] for v in physical.values()) and
                video['equal'] and motion['gait']['alternating_diagonals'] and
                motion['gait']['peak_nonfoot_ground_force_N']==0)
    evidence=dict(build='S5-r55',attempt=new,comparison_video_reviewed_on_own_F_A_r53_stage=old,
                  status='own_E30_qualified_normal_motion' if normal else 'own_E30_requires_visual_review',
                  own_per_case_original_result=new_conditions,normal_sustained_motion=normal,
                  own_started_at_t0_virgin=True,original_qualification_inherited=False,
                  recorded_500Hz_fields=arrays,recorded_physics_5kHz=physical,
                  complete_video=video,own_video=bv,stage_conditions_equal=conditions,
                  own_kinematics_and_gait=dict(moving=motion['windows'],liftoffs=motion['gait']['liftoffs'],
                       max_x_m=motion['max_x_m'],nonfoot_peak_N=motion['gait']['peak_nonfoot_ground_force_N']),
                  qualification_scope='E30 belongs to r55 with its own checkpoint, original physical starts, pass, logs, and full video. Future endpoint hold is evaluated in r55 E60.')
    write(SMALL/('r55-selection.json' if normal else 'r55-motion-comparison.json'),evidence)
    return evidence
