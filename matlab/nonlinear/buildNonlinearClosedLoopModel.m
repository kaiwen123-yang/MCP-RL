function [modelPath, simIn] = buildNonlinearClosedLoopModel(outputDir, params, nuData, wData, openModel)
%BUILDNONLINEARCLOSEDLOOPMODEL Native-block sampled feedback/plant experiment.
%   [PATH, IN] = buildNonlinearClosedLoopModel(OUTPUTDIR,P,NUDATA,WDATA,OPEN)
%   creates codesign_nonlinear_closed_loop.slx without simulating it.
%
%   P fields: k,d,tau,U,h,tStop,omega,zeta,kp,kv,ka,rho and
%             x0 = [p0;v0;a0;c0;z10;z20].
%   NUDATA/WDATA: [time,value], starting at zero, sampled every P.h.
%   The low-level feedback is computed from actual plant states in Simulink;
%   only nu, the high-level command slope, is supplied externally.
%   This does not execute an online MPC solver inside Simulink.
%
%   OUT=sim(IN) logs pLog,vLog,aLog,cLog,z1Log,z2Log,uRawLog,uSatLog,
%   wLog,nuLog as timeseries at every P.h. uRawLog is the sampled control
%   command before saturation. Control is held during each ode4 step.

arguments
    outputDir (1,1) string
    params (1,1) struct
    nuData (:,2) double
    wData (:,2) double
    openModel (1,1) logical = true
end

required = {'k','d','tau','U','h','x0','tStop', ...
    'omega','zeta','kp','kv','ka','rho'};
assert(all(isfield(params, required)), 'Codesign:ClosedLoopParameters', ...
    'Required fields: k,d,tau,U,h,x0,tStop,omega,zeta,kp,kv,ka,rho.');
validateattributes(params.x0, {'double'}, {'real','finite','numel',6});
validateattributes([params.tau,params.U,params.h,params.tStop, ...
    params.omega,params.zeta,params.kp,params.kv,params.ka,params.rho], ...
    {'double'}, {'real','finite','positive'});
params.x0 = params.x0(:);
assert(~isempty(nuData) && ~isempty(wData) && ...
    all(isfinite(nuData), 'all') && all(isfinite(wData), 'all') && ...
    nuData(1,1) == 0 && wData(1,1) == 0 && ...
    all(diff(nuData(:,1)) > 0) && all(diff(wData(:,1)) > 0), ...
    'Codesign:ClosedLoopInputs', ...
    'Input arrays must be finite, start at zero, and have increasing times.');

modelName = 'codesign_nonlinear_closed_loop';
modelPath = fullfile(outputDir, modelName + ".slx");
assert(~bdIsLoaded(modelName) && ~isfile(modelPath), ...
    'Codesign:ExistingClosedLoopModel', ...
    'The model name or output file already exists; reuse it through SimulationInput.');
if ~isfolder(outputDir)
    mkdir(outputDir);
end
load_system('simulink');
new_system(modelName);
modelWorkspace = get_param(modelName, 'ModelWorkspace');
simIn = Simulink.SimulationInput(modelName);
% Plain scalar identifiers are supported by the native Fcn expression parser.
% Prefixing prevents any dependence on similarly named base-workspace values.
for index = 1:numel(required)
    fieldName = required{index};
    variableName = ['cl_' fieldName];
    assignin(modelWorkspace, variableName, params.(fieldName));
    simIn = simIn.setVariable(variableName, params.(fieldName), 'Workspace', modelName);
end
assignin(modelWorkspace, 'nuData', nuData);
assignin(modelWorkspace, 'wData', wData);
simIn = simIn.setVariable('nuData', nuData, 'Workspace', modelName);
simIn = simIn.setVariable('wData', wData, 'Workspace', modelName);
set_param(modelName, 'SolverType', 'Fixed-step', 'Solver', 'ode4', ...
    'FixedStep', num2str(params.h,17), 'StartTime', '0', ...
    'StopTime', num2str(params.tStop,17), 'ReturnWorkspaceOutputs', 'on', ...
    'SaveTime', 'on', 'TimeSaveName', 'tout', 'SaveOutput', 'off', ...
    'SaveState', 'off', 'SignalLogging', 'off');
simIn = simIn.setModelParameter('FixedStep', num2str(params.h,17), ...
    'StopTime', num2str(params.tStop,17), 'ReturnWorkspaceOutputs', 'on');

addSource('Command slope nu', 'nuData', [30 55 130 85]);
addSource('Disturbance w', 'wData', [30 720 130 750]);

% Reference dynamics: c_dot=nu, z1_dot=z2, z2_dot=q.
addIntegrator('Command c', 4, [190 55 220 85]);
addIntegrator('Reference velocity z2', 6, [580 55 610 85]);
addIntegrator('Reference position z1', 5, [720 55 750 85]);
addFcn('Reference acceleration q', ...
    '-cl_omega^2*(u(1)-u(2))-2*cl_zeta*cl_omega*u(3)', ...
    {'Reference position z1/1','Command c/1','Reference velocity z2/1'}, ...
    [410 130 600 165]);
wire('Command slope nu/1','Command c/1');
wire('Reference acceleration q/1','Reference velocity z2/1');
wire('Reference velocity z2/1','Reference position z1/1');

% Physical plant, including actuator lag and saturation.
addIntegrator('Actuator a', 3, [850 555 880 585]);
addIntegrator('Velocity v', 2, [1130 555 1160 585]);
addIntegrator('Position p', 1, [1260 555 1290 585]);
addFcn('Nominal acceleration b', 'u(1)-cl_k*sin(u(2))-cl_d*u(3)', ...
    {'Actuator a/1','Position p/1','Velocity v/1'}, [905 640 1090 675]);
add_block('simulink/Math Operations/Sum', [modelName '/Plant acceleration'], ...
    'Inputs', '++', 'Position', [1080 550 1110 590]);
wire('Nominal acceleration b/1','Plant acceleration/1');
wire('Disturbance w/1','Plant acceleration/2');
wire('Plant acceleration/1','Velocity v/1');
wire('Velocity v/1','Position p/1');

% Feedback law split into independently inspectable expressions.
addFcn('Position error e', 'u(1)-u(2)', ...
    {'Position p/1','Reference position z1/1'}, [180 235 340 270]);
addFcn('Velocity error y', 'u(1)-u(2)', ...
    {'Velocity v/1','Reference velocity z2/1'}, [180 310 340 345]);
addFcn('Desired actuator ad', ...
    'cl_k*sin(u(1))+cl_d*u(2)+u(3)-cl_kp*u(4)-cl_kv*u(5)', ...
    {'Position p/1','Velocity v/1','Reference acceleration q/1', ...
    'Position error e/1','Velocity error y/1'}, [440 235 655 270]);
addFcn('Reference jerk qdot', ...
    '-cl_omega^2*(u(1)-u(2))-2*cl_zeta*cl_omega*u(3)', ...
    {'Reference velocity z2/1','Command slope nu/1','Reference acceleration q/1'}, ...
    [440 310 655 345]);
addFcn('Nominal desired actuator derivative', ...
    'cl_k*cos(u(1))*u(2)+(cl_d-cl_kv)*u(3)+u(4)-cl_kp*u(5)+cl_kv*u(6)', ...
    {'Position p/1','Velocity v/1','Nominal acceleration b/1', ...
    'Reference jerk qdot/1','Velocity error y/1','Reference acceleration q/1'}, ...
    [755 235 1015 275]);
addFcn('Feedback u star', ...
    'u(1)+cl_tau*(u(2)-cl_ka*(u(1)-u(3))-cl_rho*u(4))', ...
    {'Actuator a/1','Nominal desired actuator derivative/1', ...
    'Desired actuator ad/1','Velocity error y/1'}, [755 355 1015 395]);
add_block('simulink/Discrete/Zero-Order Hold', [modelName '/Sampled controller'], ...
    'SampleTime', 'cl_h', 'Position', [540 485 610 525]);
add_block('simulink/Discontinuities/Saturation', [modelName '/Input limit'], ...
    'UpperLimit', 'cl_U', 'LowerLimit', '-cl_U', 'Position', [655 485 705 525]);
addFcn('Actuator derivative', '(u(1)-u(2))/cl_tau', ...
    {'Input limit/1','Actuator a/1'}, [650 555 800 590]);
wire('Feedback u star/1','Sampled controller/1');
wire('Sampled controller/1','Input limit/1');
wire('Actuator derivative/1','Actuator a/1');

logNames = {'pLog','vLog','aLog','cLog','z1Log','z2Log', ...
    'uRawLog','uSatLog','wLog','nuLog'};
logSources = {'Position p/1','Velocity v/1','Actuator a/1','Command c/1', ...
    'Reference position z1/1','Reference velocity z2/1','Sampled controller/1', ...
    'Input limit/1','Disturbance w/1','Command slope nu/1'};
for index = 1:numel(logNames)
    y = 40 + (index-1)*68;
    add_block('simulink/Sinks/To Workspace', [modelName '/' logNames{index}], ...
        'VariableName', logNames{index}, 'SaveFormat', 'Timeseries', ...
        'MaxDataPoints', 'inf', 'Decimation', '1', 'SampleTime', 'cl_h', ...
        'Position', [1450 y 1540 y+30]);
    wire(logSources{index},[logNames{index} '/1']);
end

save_system(modelName,modelPath);
if openModel
    open_system(modelName);
end

    function addSource(name, variableName, position)
        add_block('simulink/Sources/From Workspace', [modelName '/' name], ...
            'VariableName', variableName, 'SampleTime', 'cl_h', ...
            'Interpolate', 'off', 'OutputAfterFinalValue', 'Holding final value', ...
            'Position', position);
    end

    function addIntegrator(name, stateIndex, position)
        add_block('simulink/Continuous/Integrator', [modelName '/' name], ...
            'InitialCondition', sprintf('cl_x0(%d)',stateIndex), 'Position', position);
    end

    function addFcn(name, expression, sources, position)
        muxName = [name ' inputs'];
        add_block('simulink/Signal Routing/Mux', [modelName '/' muxName], ...
            'Inputs', num2str(numel(sources)), ...
            'Position', [position(1)-40 position(2)-10 position(1)-35 position(4)+10]);
        add_block('simulink/User-Defined Functions/Fcn', [modelName '/' name], ...
            'Expr', expression, 'Position', position);
        for sourceIndex = 1:numel(sources)
            wire(sources{sourceIndex},[muxName '/' num2str(sourceIndex)]);
        end
        wire([muxName '/1'],[name '/1']);
    end

    function wire(source, destination)
        add_line(modelName,source,destination,'autorouting','on');
    end
end
