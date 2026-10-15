function [modelPath, simIn] = buildNonlinearPlantModel(outputDir, params, uData, wData, openModel)
%BUILDNONLINEARPLANTMODEL Build a native-block nonlinear plant replay model.
%   [PATH, IN] = buildNonlinearPlantModel(OUTPUTDIR, P, UDATA, WDATA, OPEN)
%   builds codesign_nonlinear_plant.slx without running a simulation.
%
%   Plant: p_dot = v; v_dot = a - k*sin(p) - d*v + w;
%          tau*a_dot = -a + sat_U(u).
%   P fields: k, d, tau > 0, U > 0, h > 0, x0 = [p0; v0; a0], tStop.
%   UDATA and WDATA are [time, value] arrays. Inputs are sampled every P.h
%   and held throughout each ode4 integration step. Start their data at t=0
%   and align command/disturbance changes to the P.h sampling grid.
%
%   OUT = sim(IN) returns pLog, vLog, aLog, uLog, uSatLog, wLog as timeseries.
%   OPEN defaults to true. The function does not close or clear any model.
%   An existing file or loaded model with this name is deliberately preserved.
%
%   This model independently replays recorded u/w inputs; it contains no
%   controller, optimizer, learned policy, or MATLAB Function plant wrapper.

arguments
    outputDir (1,1) string
    params (1,1) struct
    uData (:,2) double
    wData (:,2) double
    openModel (1,1) logical = true
end

required = {'k','d','tau','U','h','x0','tStop'};
assert(all(isfield(params, required)), 'Codesign:PlantParameters', ...
    'params needs k, d, tau, U, h, x0, and tStop.');
validateattributes(params.x0, {'double'}, {'real','finite','numel',3});
validateattributes([params.tau, params.U, params.h, params.tStop], ...
    {'double'}, {'real','finite','positive'});
validateattributes([params.k, params.d], {'double'}, {'real','finite'});
params.x0 = params.x0(:);
assert(~isempty(uData) && ~isempty(wData) && ...
    all(isfinite(uData), 'all') && all(isfinite(wData), 'all') && ...
    uData(1,1) == 0 && wData(1,1) == 0 && ...
    all(diff(uData(:,1)) > 0) && all(diff(wData(:,1)) > 0), ...
    'Codesign:PlantInputs', ...
    'Each input must be finite [time,value] data, starting at 0 with increasing times.');

modelName = 'codesign_nonlinear_plant';
modelPath = fullfile(outputDir, modelName + ".slx");
assert(~bdIsLoaded(modelName) && ~isfile(modelPath), ...
    'Codesign:ExistingModel', ...
    'The model name or output file already exists; reuse it through SimulationInput.');
if ~isfolder(outputDir)
    mkdir(outputDir);
end

load_system('simulink');
new_system(modelName);
modelWorkspace = get_param(modelName, 'ModelWorkspace');
assignin(modelWorkspace, 'plantParams', params);
assignin(modelWorkspace, 'uData', uData);
assignin(modelWorkspace, 'wData', wData);

% Solver settings are set from P and mirrored in the returned SimulationInput.
% Numeric strings avoid relying on model-workspace lookup for config settings.
set_param(modelName, 'SolverType', 'Fixed-step', 'Solver', 'ode4', ...
    'FixedStep', num2str(params.h, 17), 'StartTime', '0', ...
    'StopTime', num2str(params.tStop, 17), ...
    'ReturnWorkspaceOutputs', 'on', 'SaveTime', 'on', ...
    'TimeSaveName', 'tout', 'SaveOutput', 'off', 'SaveState', 'off', ...
    'SignalLogging', 'off');

add_block('simulink/Sources/From Workspace', [modelName '/Command'], ...
    'VariableName', 'uData', 'SampleTime', 'plantParams.h', ...
    'Interpolate', 'off', 'OutputAfterFinalValue', 'Holding final value', ...
    'Position', [30 70 130 100]);
add_block('simulink/Discontinuities/Saturation', [modelName '/Input limit'], ...
    'UpperLimit', 'plantParams.U', 'LowerLimit', '-plantParams.U', ...
    'Position', [180 65 230 105]);
add_block('simulink/Math Operations/Sum', [modelName '/Actuator balance'], ...
    'Inputs', '+-', 'Position', [280 65 310 105]);
add_block('simulink/Math Operations/Gain', [modelName '/Inverse time constant'], ...
    'Gain', '1/plantParams.tau', 'Position', [350 65 410 105]);
add_block('simulink/Continuous/Integrator', [modelName '/Actuator state a'], ...
    'InitialCondition', 'plantParams.x0(3)', 'Position', [460 65 490 105]);
add_block('simulink/Math Operations/Sum', [modelName '/Acceleration balance'], ...
    'Inputs', '+--+', 'Position', [590 60 625 120]);
add_block('simulink/Continuous/Integrator', [modelName '/Velocity v'], ...
    'InitialCondition', 'plantParams.x0(2)', 'Position', [690 65 720 105]);
add_block('simulink/Continuous/Integrator', [modelName '/Position p'], ...
    'InitialCondition', 'plantParams.x0(1)', 'Position', [800 65 830 105]);

add_block('simulink/Math Operations/Trigonometric Function', [modelName '/Sine'], ...
    'Operator', 'sin', 'Position', [795 235 845 270]);
add_block('simulink/Math Operations/Gain', [modelName '/Nonlinear restoring force'], ...
    'Gain', 'plantParams.k', 'Position', [670 235 730 270], ...
    'Orientation', 'left');
add_block('simulink/Math Operations/Gain', [modelName '/Viscous damping'], ...
    'Gain', 'plantParams.d', 'Position', [670 165 730 200], ...
    'Orientation', 'left');
add_block('simulink/Sources/From Workspace', [modelName '/Disturbance'], ...
    'VariableName', 'wData', 'SampleTime', 'plantParams.h', ...
    'Interpolate', 'off', 'OutputAfterFinalValue', 'Holding final value', ...
    'Position', [405 330 505 360]);

add_line(modelName, 'Command/1', 'Input limit/1', 'autorouting', 'on');
add_line(modelName, 'Input limit/1', 'Actuator balance/1', 'autorouting', 'on');
add_line(modelName, 'Actuator balance/1', 'Inverse time constant/1', 'autorouting', 'on');
add_line(modelName, 'Inverse time constant/1', 'Actuator state a/1', 'autorouting', 'on');
add_line(modelName, 'Actuator state a/1', 'Actuator balance/2', 'autorouting', 'on');
add_line(modelName, 'Actuator state a/1', 'Acceleration balance/1', 'autorouting', 'on');
add_line(modelName, 'Acceleration balance/1', 'Velocity v/1', 'autorouting', 'on');
add_line(modelName, 'Velocity v/1', 'Position p/1', 'autorouting', 'on');
add_line(modelName, 'Position p/1', 'Sine/1', 'autorouting', 'on');
add_line(modelName, 'Sine/1', 'Nonlinear restoring force/1', 'autorouting', 'on');
add_line(modelName, 'Nonlinear restoring force/1', 'Acceleration balance/2', 'autorouting', 'on');
add_line(modelName, 'Velocity v/1', 'Viscous damping/1', 'autorouting', 'on');
add_line(modelName, 'Viscous damping/1', 'Acceleration balance/3', 'autorouting', 'on');
add_line(modelName, 'Disturbance/1', 'Acceleration balance/4', 'autorouting', 'on');

logNames = {'pLog','vLog','aLog','uLog','uSatLog','wLog'};
logSources = {'Position p/1','Velocity v/1','Actuator state a/1', ...
    'Command/1','Input limit/1','Disturbance/1'};
for index = 1:numel(logNames)
    y = 55 + (index - 1) * 65;
    add_block('simulink/Sinks/To Workspace', [modelName '/' logNames{index}], ...
        'VariableName', logNames{index}, 'SaveFormat', 'Timeseries', ...
        'MaxDataPoints', 'inf', 'Decimation', '1', 'SampleTime', 'plantParams.h', ...
        'Position', [950 y 1040 y+30]);
    add_line(modelName, logSources{index}, [logNames{index} '/1'], 'autorouting', 'on');
end

save_system(modelName, modelPath);
simIn = Simulink.SimulationInput(modelName);
simIn = simIn.setVariable('plantParams', params, 'Workspace', modelName);
simIn = simIn.setVariable('uData', uData, 'Workspace', modelName);
simIn = simIn.setVariable('wData', wData, 'Workspace', modelName);
simIn = simIn.setModelParameter('FixedStep', num2str(params.h, 17), ...
    'StopTime', num2str(params.tStop, 17), 'ReturnWorkspaceOutputs', 'on');
if openModel
    open_system(modelName);
end
end
