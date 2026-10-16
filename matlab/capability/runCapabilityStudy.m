function result = runCapabilityStudy(root)
%RUNCAPABILITYSTUDY Frozen classical mechanism probes; no RL training.
if nargin==0
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
addpath(fullfile(root,'matlab','capability'));
publicDir = fullfile(root,'notes','capability-probe-2026-10-16');
cacheDir = fullfile(root,'work','capability-probe-2026-10-16');
if ~exist(cacheDir,'dir'), mkdir(cacheDir); end
wholeClock = tic;
kpList = [1 3 6]; kdList = [0.5 1.5]; nonlinList = [0 3 8];
grid = zeros(18,3); designCost = zeros(18,2); cursor = 0;
designRuns = cell(18,2);
fprintf('Frozen design: 18 feedbacks x 2 tasks.\n');
for kp = kpList
    for kd = kdList
        for nl = nonlinList
            cursor = cursor+1;
            grid(cursor,:) = [kp kd nl];
            for task = 1:2
                cfg = capabilityConfig(task);
                designRuns{cursor,task} = capabilitySimulate(grid(cursor,:),cfg);
                designCost(cursor,task) = designRuns{cursor,task}.metrics.cost;
            end
            fprintf('%02d: [%g %g %g], design mean %.6f\n',cursor,kp,kd,nl,mean(designCost(cursor,:)));
        end
    end
end
meanCost = mean(designCost,2);
linearIndices = find(grid(:,3)==0);
[~,localBest] = min(meanCost(linearIndices));
linearIndex = linearIndices(localBest);
[~,extendedIndex] = min(meanCost);
selectedLinear = grid(linearIndex,:);
selectedExtended = grid(extendedIndex,:);
fprintf('Selected linear [%g %g %g], extended [%g %g %g].\n',selectedLinear,selectedExtended);
evaluation = cell(3,4);
labels = {'linear_current_mpc','extended_current_mpc', ...
    'linear_fast_command_mpc','extended_replay_linear_commands'};
rows = cell(3,4);
for k = 1:3
    caseId = k+2;
    cfg = capabilityConfig(caseId);
    evaluation{k,1} = capabilitySimulate(selectedLinear,cfg);
    evaluation{k,2} = capabilitySimulate(selectedExtended,cfg);
    fastCfg = cfg; fastCfg.planEvery = 1;
    evaluation{k,3} = capabilitySimulate(selectedLinear,fastCfg);
    evaluation{k,4} = capabilitySimulate(selectedExtended,cfg,evaluation{k,1}.command);
    for g = 1:4
        rr = evaluation{k,g};
        item = rr.metrics;
        item.caseId = caseId;
        item.group = labels{g};
        item.feedback = rr.theta;
        rows{k,g} = item;
        fprintf('Evaluation case%d %s cost %.6f\n',caseId,labels{g},item.cost);
    end
end
rows = [rows{:}];
probes = cell(2,3);
for mode = 1:2
    cfg = capabilityConfig(99);
    if mode==2, cfg.s0(2)=0.22; cfg.s0(4)=0.25; end
    fixedCommand = 0.65*ones(round(cfg.T/cfg.dt),1);
    for j = 1:3
        probes{mode,j} = capabilitySimulate([3 0.5 nonlinList(j)],cfg,fixedCommand);
    end
end
save(fullfile(cacheDir,'raw-checkpoint.mat'),'grid','designCost', ...
    'designRuns','evaluation','probes','rows','selectedLinear','selectedExtended');
audit = mathAudit(probes);
design = struct('grid',grid,'perTaskCost',designCost, ...
    'meanCost',meanCost,'selectedLinear',selectedLinear, ...
    'selectedExtended',selectedExtended, ...
    'selectionUsesCases',[1 2],'evaluationCases',[3 4 5]);
allMetrics = cellfun(@(z) z.metrics,designRuns,'UniformOutput',false);
allMetrics = [allMetrics(:);cellfun(@(z) z.metrics,evaluation(:),'UniformOutput',false); ...
    cellfun(@(z) z.metrics,probes(:),'UniformOutput',false)];
ledger = struct('runs',numel(allMetrics),'physicalTransitions',0, ...
    'mpcCalls',0,'candidateTransitions',0,'solverSecondsTotal',0);
for i=1:numel(allMetrics)
    mm=allMetrics{i};
    ledger.physicalTransitions=ledger.physicalTransitions+mm.physicalTransitions;
    ledger.mpcCalls=ledger.mpcCalls+mm.mpcCalls;
    ledger.candidateTransitions=ledger.candidateTransitions+mm.candidateTransitions;
    ledger.solverSecondsTotal=ledger.solverSecondsTotal+mm.solverSecondsTotal;
end
ledger.studyWallSeconds = toc(wholeClock);
result = struct('date','2026-10-16','matlabVersion',version, ...
    'identity','Known-model classical feedback mechanism study, no RL', ...
    'design',design,'evaluation',rows,'mathAudit',audit,'ledger',ledger, ...
    'trainingRun',false,'simulinkStatus','not_yet_run');
save(fullfile(cacheDir,'study.mat'),'result','designRuns','evaluation','probes');
writeJson(fullfile(publicDir,'results.json'),result);
writetable(struct2table(rows),fullfile(publicDir,'evaluation.csv'));
writematrix([grid designCost meanCost],fullfile(publicDir,'design-grid.csv'));
fprintf('Completed mechanism study in %.2fs.\n',ledger.studyWallSeconds);
end

function audit = mathAudit(probes)
cfg = capabilityConfig(99);
x = [0;0.2;0.6;0.6;0;0]; c = 0.8; base = [3 0.5 0];
changes = [0 1 3]; entries = cell(1,3);
for j=1:numel(changes)
    theta = base; theta(3)=changes(j);
    [raw,f,g] = capabilityControlLaw(x,c,theta,cfg);
    allowance = cfg.U-abs(g);
    cLo = (-allowance-(cfg.dv-cfg.kv)*x(3))/cfg.kv;
    cHi = (allowance-(cfg.dv-cfg.kv)*x(3))/cfg.kv;
    entries{j} = struct('nonlinearDamping',changes(j),'f',f,'g',g, ...
        'resourceDemand',abs(f)+abs(g),'rawPeak',max(abs(raw)), ...
        'currentCommandFeasible',max(abs(raw))<=cfg.U, ...
        'anyCommandFeasible',allowance>=0 && max(cLo,cfg.cMin)<=min(cHi,cfg.cMax), ...
        'commandInterval',[max(cLo,cfg.cMin),min(cHi,cfg.cMax)]);
end
entries = [entries{:}];
% At this state theta=1 is feasible at c=.6, but cannot be matched by
% theta=0 even when the latter scans saturated commands of any amplitude.
[newRaw,~,newG] = capabilityControlLaw(x,0.6,[3 0.5 1],cfg);
[~,~,oldG] = capabilityControlLaw(x,0.6,base,cfg);
cScan = linspace(-10,10,20001);
oldRaw = capabilityControlLaw(repmat(x,1,numel(cScan)),cScan,base,cfg);
oldAction = min(cfg.U,max(-cfg.U,oldRaw));
residual = vecnorm(oldAction-newRaw,2,1);
[minimumResidual,ix] = min(residual);
nominalDiff = zeros(1,2);
for j=2:3
    nominalDiff(j-1)=max(abs(probes{1,j}.x-probes{1,1}.x),[],'all');
end
audit = struct('resourceEntries',entries,'nonreplicableState',x', ...
    'newCommand',0.6,'newTheta',[3 0.5 1],'newRawAction',newRaw', ...
    'newActionUnsaturated',max(abs(newRaw))<cfg.U, ...
    'oldDifferentialLimit',2*abs(oldG),'newDifferentialMagnitude',2*abs(newG), ...
    'scanMinActionResidual',minimumResidual,'scanBestCommand',cScan(ix), ...
    'analyticUnsaturatedResidual',sqrt(2)*abs(newG-oldG), ...
    'nominalResponseMaxDiff',nominalDiff,'smallSignalNonlinearTermOrder',3, ...
    'scope','Instantaneous action non-equivalence; not a global task-optimality result');
end

function writeJson(path,value)
fid = fopen(path,'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
