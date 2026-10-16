function cfg = capabilityConfig(caseId)
%CAPABILITYCONFIG Frozen, no-learning mechanism-probe settings.
if nargin == 0
    caseId = 1;
end
cfg = struct('dt',0.02,'integrationStep',0.005,'macroEvery',5, ...
    'planEvery',5,'H',12,'T',6,'U',1.8,'dv',0.2,'k',1, ...
    'dr',0.15,'tau',[0.1;0.1],'kv',3,'q0',0.15, ...
    'cMin',-0.2,'cMax',1.1,'qMax',0.35, ...
    'nCandidates',64,'nIterations',3,'nElite',8, ...
    'caseId',caseId,'s0',zeros(6,1),'seed',71016);
cfg.weights = [0.3 18 5 0.6 0.025]; % p,q,v,r,input energy
if caseId == 2
    cfg.s0(2) = 0.22;
    cfg.s0(4) = -0.12;
elseif caseId == 4
    cfg.tau = [0.08;0.12];
elseif caseId == 5
    cfg.s0(2) = -0.18;
    cfg.s0(4) = 0.15;
end
end
