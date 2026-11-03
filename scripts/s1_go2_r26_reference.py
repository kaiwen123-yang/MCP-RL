"""W37 local base reference; userdata84:96 is frozen until the next replan."""
import numpy as np

PACKET_FIELDS=('active','t0','x_anchor','endpoint','direction','v_anchor','cap',
               'heading','world_head_dx','world_head_dy','base_y','base_z')


def reference_packet(mission,t,x_anchor,cap,offset,settings):
    active=mission.phase in ('outbound','return')
    far=mission.phase in ('outbound','far_hold','turn')
    endpoint=float(mission.task['turnaround_xy_m' if far else 'return_xy_m'][0])
    direction=float(np.sign(endpoint-x_anchor))
    velocity=float(np.clip(direction*mission.v[0],0.,settings['maximum_speed_m_s']))
    heading=float(mission.heading)
    co,si=np.cos(heading),np.sin(heading)
    dx=co*offset[0]-si*offset[1];dy=si*offset[0]+co*offset[1]
    return np.array([active,t,x_anchor,endpoint,direction,velocity,cap,heading,
                     dx,dy,0.,mission.body_target[2]],dtype=float)


def base_reference(packet,t,settings):
    # Same analytic integral and clamping as mjpc/r26_reference.h.
    p=packet
    tau=float(np.clip(t-p[1],0.,settings['horizon_s']))
    lead=max(0.,settings['maximum_lead_m']/settings['maximum_speed_m_s']-settings['horizon_s'])
    dv=p[6]-p[5];a=settings['acceleration_m_s2'];b=min(tau,abs(dv)/a)
    integral=p[5]*b+.5*np.sign(dv)*a*b*b+p[6]*(tau-b)
    x=p[2]+p[4]*min(abs(p[3]-p[2]),p[5]*lead+integral)
    return np.array([x,p[10],p[11]])
