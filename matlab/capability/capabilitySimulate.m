function run = capabilitySimulate(theta,cfg,commandReplay)
%CAPABILITYSIMULATE Continuous plant approximation and online sampled MPC.
if nargin<3
    commandReplay = [];
end
n = round(cfg.T/cfg.dt);
t = (0:n)'*cfg.dt;
x = zeros(n+1,6);
x(1,:) = cfg.s0';
raw = zeros(n,2);
u = raw;
c = zeros(n,1);
w = c;
stage = c;
plan = zeros(cfg.H,1);
solveTime = zeros(n,1);
replanned = false(n,1);
feasible = true(n,1);
candidateTransitions = 0;
for i = 1:n
    if isempty(commandReplay)
        [uk,c(i),plan,info] = capabilityController(x(i,:)',t(i),theta,cfg,plan);
        replanned(i) = info.replanned;
        feasible(i) = info.feasible;
        solveTime(i) = info.solveSeconds;
        candidateTransitions = candidateTransitions+info.candidateTransitions;
    else
        c(i) = commandReplay(i);
        uk = capabilityControlLaw(x(i,:)',c(i),theta,cfg);
    end
    raw(i,:) = uk';
    u(i,:) = min(cfg.U,max(-cfg.U,uk))';
    w(i) = capabilityDisturbance(t(i),cfg);
    stage(i) = capabilityStageCost(x(i,:)',u(i,:)',t(i),cfg);
    x(i+1,:) = capabilityIntegrate(x(i,:)',u(i,:)',w(i),cfg, ...
        round(cfg.dt/cfg.integrationStep))';
end
run = struct('theta',theta,'cfg',cfg,'t',t,'x',x,'raw',raw, ...
    'u',u,'command',c,'disturbance',w,'stageCost',stage);
run.metrics = struct('cost',cfg.dt*sum(stage), ...
    'velocityRMSE',sqrt(mean((x(1:n,3)-referenceVelocity(t(1:n))).^2)), ...
    'angleRMS',sqrt(mean(x(1:n,2).^2)), ...
    'anglePeak',max(abs(x(:,2))),'terminalPosition',x(end,1), ...
    'inputPeak',max(abs(u),[],'all'),'rawInputPeak',max(abs(raw),[],'all'), ...
    'inputEnergy',cfg.dt*sum(u.^2,'all'), ...
    'saturationFraction',mean(any(abs(raw)>cfg.U+1e-9,2)), ...
    'angleViolationFraction',mean(abs(x(1:n,2))>cfg.qMax), ...
    'mpcCalls',sum(replanned),'noFeasibleSample',sum(replanned & ~feasible), ...
    'solverSecondsTotal',sum(solveTime),'solverSecondsMax',max(solveTime), ...
    'physicalTransitions',n,'candidateTransitions',candidateTransitions);
end

function v = referenceVelocity(t)
[~,v] = capabilityReference(t);
end
