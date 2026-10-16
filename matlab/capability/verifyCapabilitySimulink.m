function checks = verifyCapabilitySimulink(root)
%VERIFYCAPABILITYSIMULINK Run native plant plus live MPC for both selections.
if nargin==0
    root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
cacheDir=fullfile(root,'work','capability-probe-2026-10-16');
publicDir=fullfile(root,'notes','capability-probe-2026-10-16');
data=load(fullfile(cacheDir,'study.mat'),'result','evaluation');
checkCells=cell(1,2);
models=cell(1,2);
for j=1:2
    reference=data.evaluation{1,j};
    [modelPath,simIn,modelName]=buildCapabilityModel(string(root), ...
        reference.cfg,reference.theta);
    timer=tic;
    simOut=sim(simIn);
    elapsed=toc(timer);
    states=squeeze(simOut.stateLog.Data);
    raw=squeeze(simOut.rawLog.Data);
    input=squeeze(simOut.uLog.Data);
    command=simOut.commandLog.Data(:);
    disturbance=simOut.disturbanceLog.Data(:);
    counts=squeeze(simOut.controllerCountLog.Data);
    n=size(reference.u,1);
    ck=struct('group',data.result.evaluation(j*3-2).group, ...
        'caseId',3,'theta',reference.theta,'stateSamples',size(states,1), ...
        'stateMaxAbsDiff',max(abs(states-reference.x),[],'all'), ...
        'rawMaxAbsDiff',max(abs(raw(1:n,:)-reference.raw),[],'all'), ...
        'inputMaxAbsDiff',max(abs(input(1:n,:)-reference.u),[],'all'), ...
        'commandMaxAbsDiff',max(abs(command(1:n)-reference.command)), ...
        'disturbanceMaxAbsDiff',max(abs(disturbance(1:n)-reference.disturbance)), ...
        'timeMaxAbsDiff',max(abs(simOut.stateLog.Time-reference.t)), ...
        'controllerCalls',counts(end,1),'actualOnlineMPCCalls',counts(end,2), ...
        'expectedOnlineMPCCalls',reference.metrics.mpcCalls, ...
        'wallSeconds',elapsed,'modelFile',char(fullfile( ...
        'work','capability-probe-2026-10-16',modelName+".slx")));
    ck.passed=ck.stateMaxAbsDiff<1e-8 && ck.inputMaxAbsDiff<1e-8 && ...
        ck.commandMaxAbsDiff<1e-8 && ck.disturbanceMaxAbsDiff<1e-12 && ...
        ck.actualOnlineMPCCalls==ck.expectedOnlineMPCCalls;
    checkCells{j}=ck;
    models{j}=struct('name',modelName,'path',modelPath);
    save(fullfile(cacheDir,sprintf('simulink-group%d.mat',j)),'simOut','ck');
    fprintf('%s: state error %.3g, live MPC calls %d, passed %d\n', ...
        ck.group,ck.stateMaxAbsDiff,ck.actualOnlineMPCCalls,ck.passed);
end
checks=struct('date','2026-10-16','matlabVersion',version, ...
    'nativeContinuousPlant',true,'onlineMPC',true,'commandReplay',false, ...
    'checks',[checkCells{:}],'allPassed',all(cellfun(@(z)z.passed,checkCells)));
fid=fopen(fullfile(publicDir,'simulink-check.json'),'w');
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(checks,'PrettyPrint',true));
result=data.result;
if checks.allPassed
    result.simulinkStatus='passed_two_live_MPC_native_plant_comparisons';
else
    result.simulinkStatus='comparison_failed_inspect_simulink_check';
end
result.simulinkEvidence='notes/capability-probe-2026-10-16/simulink-check.json';
save(fullfile(cacheDir,'study.mat'),'result','-append');
resultFile=fopen(fullfile(publicDir,'results.json'),'w');
resultCleanup=onCleanup(@()fclose(resultFile));
fprintf(resultFile,'%s\n',jsonencode(result,'PrettyPrint',true));
save(fullfile(cacheDir,'model-locations.mat'),'models');
% Show the most recent complete online model in the user's MATLAB desktop.
open_system(models{end}.name);
end
