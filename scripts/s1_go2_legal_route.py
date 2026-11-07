"""Shared 500Hz route account for rewards and independent evaluation."""
import numpy as np


class LegalRoute:
    """Reward account on the original 500Hz strip passage criterion.

    Approach/crossing credit is provisional. A side exit or an uncompleted
    passage retracts that lap's positive credit and disables further positive
    progress until the next natural lap. Physics and reward settlement are unchanged.
    """
    def __init__(self):
        self.progress=0.;self.lap_credit=0.;self.completed_laps=0
        self.raw_laps=0;self.valid=True;self.previous_phase=None;self.previous_x=None
        self.passages=[];self.current=None;self.leg_pass={}
        self.side_exits=0

    def invalidate(self):
        if self.valid:
            retract=max(self.lap_credit,0.)
            self.progress-=retract;self.lap_credit-=retract
        self.valid=False

    def update(self,t,mission,xy,bottom,grf):
        phase=mission.phase
        raw_laps=getattr(mission,'laps',int(phase=='complete'))
        if self.previous_phase in ('outbound','return'):
            direction=1 if self.previous_phase=='outbound' else -1
            if self.valid:
                change=direction*(float(xy[0])-self.previous_x)/6
                self.progress+=change;self.lap_credit+=change
        # Close the actual leg when the mission changes to its endpoint/turn.
        if self.current is not None and phase!=self.current['direction']:
            self.leg_pass[self.current['direction']]=self.current['completed']
            if not self.current['completed']:self.invalidate()
            self.current=None
        if raw_laps>self.raw_laps:
            self.completed_laps+=int(self.valid and self.leg_pass.get('outbound',False)
                                     and self.leg_pass.get('return',False))
            self.raw_laps=raw_laps
            self.lap_credit=0.;self.valid=True;self.leg_pass={}
        if phase in ('outbound','return'):
            if self.current is None:
                self.current=dict(lap=raw_laps+1,direction=phase,entry_s=None,exit_s=None,
                    lateral_exit_s=None,material_loaded=False,completed=False)
                self.passages.append(self.current)
            row=self.current
            enter=xy[0]>=.8 if phase=='outbound' else xy[0]<=2.2
            leave=xy[0]>=2.2 if phase=='outbound' else xy[0]<=.8
            if row['entry_s'] is None and enter:row['entry_s']=float(t)
            if row['entry_s'] is not None and row['exit_s'] is None:
                if abs(xy[1])>.6 and row['lateral_exit_s'] is None:
                    row['lateral_exit_s']=float(t);self.side_exits+=1;self.invalidate()
                half=.015811388300841896
                inside=(bottom[:,0]>=.8+half)&(bottom[:,0]<=2.2-half)&(abs(bottom[:,1])<=.6-half)
                row['material_loaded'] |= bool(np.any(inside&(grf[:,2]>0)))
                if leave:
                    row['exit_s']=float(t)
                    row['completed']=row['lateral_exit_s'] is None and row['material_loaded']
                    if not row['completed']:self.invalidate()
        self.previous_x=float(xy[0]);self.previous_phase=phase

    def settle_physical_terminal(self,t):
        before=self.progress
        if self.current is not None and not self.current['completed']:
            self.invalidate()
            self.current['terminal_closed_s']=float(t)
        return dict(progress_before=before,progress_after=self.progress,
                    retracted_positive_credit=before-self.progress,
                    reason='true_physical_terminal')

    def result(self):
        return dict(reward_version='W36_legal_route_progress_and_lap_alignment',
            progress=self.progress,completed_roundtrips=self.completed_laps,
            legal_outbound_traversals=sum(x['direction']=='outbound' and x['completed'] for x in self.passages),
            legal_return_traversals=sum(x['direction']=='return' and x['completed'] for x in self.passages),
            side_exits=self.side_exits,raw_roundtrips=self.raw_laps,passages=self.passages,
            criterion='Original base strip x[.8,2.2],abs(y)<=.6 and actual interior-foot Fz>0 at500Hz; return uses reversed boundaries. Signed credit is provisional and retracted on illegal/incomplete passage. New R12 mission evaluation also checks these recorded legal roundtrips.')

