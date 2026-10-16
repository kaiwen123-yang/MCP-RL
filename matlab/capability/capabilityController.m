function [raw,c,plan,info] = capabilityController(x,t,theta,cfg,plan)
%CAPABILITYCONTROLLER Online planner plus fast state-feedback execution.
sample = round(t/cfg.dt);
info = struct('replanned',false,'feasible',true,'nominalCost',NaN, ...
    'maxRaw',NaN,'maxAngle',NaN,'candidateTransitions',0,'solveSeconds',0);
if mod(sample,cfg.planEvery)==0 && t<cfg.T-1e-10
    if sample>0
        % Project the time-shifted previous piecewise-constant plan onto the
        % new prediction grid. Fast replanning shifts by dt, not a full block.
        fraction = cfg.planEvery/cfg.macroEvery;
        warm = (1-fraction)*plan+fraction*[plan(2:end);plan(end)];
    else
        warm = plan;
    end
    solveClock = tic;
    [plan,mpcInfo] = capabilityMPC(x,t,theta,cfg,warm);
    info = mpcInfo;
    info.solveSeconds = toc(solveClock);
    info.replanned = true;
end
c = plan(1);
raw = capabilityControlLaw(x,c,theta,cfg);
end
