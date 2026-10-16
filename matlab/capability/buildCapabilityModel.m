function [modelPath, simIn, modelName] = buildCapabilityModel(root, cfg, theta)
%BUILDCAPABILITYMODEL Build the online MPC/feedback/native physical plant.
%   [PATH,IN,NAME] = buildCapabilityModel(ROOT,CFG,THETA) creates an SLX in
%   ROOT/work/capability-probe-2026-10-16. Run OUT=sim(IN) to execute it.
%   The builder never closes an existing model and does not simulate.
%   All model parameters are stored in the model workspace and IN.
%
%   Actual state s=[p;q;v;r;a1;a2] feeds capabilityController every cfg.dt.
%   That function decides when to replan using cfg.planEvery.
%   The physical dynamics use native Fcn/Integrator/Saturation blocks.
%   No measured states, planner commands, or actuator actions are replayed.
%
%   OUT logs stateLog,rawLog,uLog,commandLog,disturbanceLog and
%   controllerCountLog as timeseries at cfg.dt, including both endpoints.

arguments
    root (1,1) string
    cfg (1,1) struct
    theta (1,3) double
end
assert(abs(cfg.dt / cfg.integrationStep - round(cfg.dt / cfg.integrationStep)) ...
    < 1e-10, 'Capability:Sampling', ...
    'cfg.dt must be an integer multiple of cfg.integrationStep.');
if isfield(cfg, 's0')
    initialState = cfg.s0(:);
else
    initialState = zeros(6, 1);
end
validateattributes(initialState, {'double'}, {'real','finite','numel',6});
outputDirectory = fullfile(root, 'work', 'capability-probe-2026-10-16');
if ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end
modelName = 'capability_online_probe';
modelPath = fullfile(outputDirectory, modelName + ".slx");
if bdIsLoaded(modelName) || isfile(modelPath)
    modelName = ['capability_probe_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmssSSS'))];
    modelPath = fullfile(outputDirectory, modelName + ".slx");
end
load_system('simulink');
new_system(modelName);
modelWorkspace = get_param(modelName, 'ModelWorkspace');
simIn = Simulink.SimulationInput(modelName);
variables = struct('cap_cfg', cfg, 'cap_theta', theta, ...
    'cap_dt', cfg.dt, 'cap_U', cfg.U, 'cap_dv', cfg.dv, ...
    'cap_k', cfg.k, 'cap_dr', cfg.dr, ...
    'cap_tau1', cfg.tau(1), 'cap_tau2', cfg.tau(2), 'cap_s0', initialState);
variableNames = fieldnames(variables);
for index = 1:numel(variableNames)
    name = variableNames{index};
    value = variables.(name);
    assignin(modelWorkspace, name, value);
    simIn = simIn.setVariable(name, value, 'Workspace', modelName);
end
set_param(modelName, 'SolverType', 'Fixed-step', 'Solver', 'ode4', ...
    'FixedStep', num2str(cfg.integrationStep, 17), 'StartTime', '0', ...
    'StopTime', num2str(cfg.T, 17), 'ReturnWorkspaceOutputs', 'on', ...
    'SaveTime', 'on', 'TimeSaveName', 'tout', 'SaveOutput', 'off', ...
    'SaveState', 'off', 'SignalLogging', 'off');
simIn = simIn.setModelParameter('FixedStep', num2str(cfg.integrationStep, 17), ...
    'StopTime', num2str(cfg.T, 17), 'ReturnWorkspaceOutputs', 'on');

% Model-level control/plant graph. The disturbance is sampled with feedback.
add_block('simulink/Discrete/Zero-Order Hold', [modelName '/Sample actual state'], ...
    'SampleTime', 'cap_dt', 'Position', [90 160 160 205]);
add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', ...
    [modelName '/Online MPC and feedback'], ...
    'FunctionName', 'capabilityControllerBlock', 'Parameters', 'cap_cfg,cap_theta', ...
    'Position', [215 130 405 240]);
add_block('simulink/Discontinuities/Saturation', [modelName '/Actuator input limits'], ...
    'UpperLimit', 'cap_U', 'LowerLimit', '-cap_U', ...
    'Position', [470 120 550 160]);
plant = [modelName '/Physical plant'];
add_block('simulink/Ports & Subsystems/Subsystem', plant, ...
    'Position', [640 125 800 220]);
Simulink.SubSystem.deleteContents(plant);
buildPlant(plant);
wire(modelName, 'Sample actual state/1', 'Online MPC and feedback/1');
wire(modelName, 'Online MPC and feedback/1', 'Actuator input limits/1');
wire(modelName, 'Actuator input limits/1', 'Physical plant/1');
wire(modelName, 'Online MPC and feedback/3', 'Physical plant/2');
wire(modelName, 'Physical plant/1', 'Sample actual state/1');

names = {'stateLog','rawLog','uLog','commandLog','disturbanceLog','controllerCountLog'};
sources = {'Physical plant/1', 'Online MPC and feedback/1', ...
    'Actuator input limits/1', 'Online MPC and feedback/2', ...
    'Online MPC and feedback/3', 'Online MPC and feedback/4'};
for index = 1:numel(names)
    y = 40 + (index - 1) * 65;
    add_block('simulink/Sinks/To Workspace', [modelName '/' names{index}], ...
        'VariableName', names{index}, 'SaveFormat', 'Timeseries', ...
        'MaxDataPoints', 'inf', 'Decimation', '1', 'SampleTime', 'cap_dt', ...
        'Position', [925 y 1085 y + 30]);
    wire(modelName, sources{index}, [names{index} '/1']);
end
annotation = Simulink.Annotation(modelName, ...
    ['Real physical state -> sampled feedback -> online MPC / feedback -> saturated actuator input.' newline ...
    'Online macro replanning uses the current Simulink state; no command or action playback.' newline ...
    'Open Physical plant to inspect the six continuous state equations.']);
annotation.Position = [70 415];
save_system(modelName, modelPath);
end

function buildPlant(plant)
add_block('simulink/Ports & Subsystems/In1', [plant '/Applied input u'], ...
    'Port', '1', 'PortDimensions', '2', 'Position', [45 55 75 75]);
add_block('simulink/Ports & Subsystems/In1', [plant '/Held disturbance w'], ...
    'Port', '2', 'PortDimensions', '1', 'Position', [45 115 75 135]);
add_block('simulink/Continuous/Integrator', [plant '/Actual physical state'], ...
    'InitialCondition', 'cap_s0', 'Position', [740 205 790 250]);
add_block('simulink/Ports & Subsystems/Out1', [plant '/State s'], ...
    'Port', '1', 'Position', [890 220 920 240]);
add_block('simulink/Signal Routing/Mux', [plant '/State input disturbance'], ...
    'Inputs', '3', 'Position', [170 170 175 290]);
add_block('simulink/Signal Routing/Mux', [plant '/State derivatives'], ...
    'Inputs', '6', 'Position', [650 80 655 380]);
wire(plant, 'Actual physical state/1', 'State input disturbance/1');
wire(plant, 'Applied input u/1', 'State input disturbance/2');
wire(plant, 'Held disturbance w/1', 'State input disturbance/3');
wire(plant, 'State derivatives/1', 'Actual physical state/1');
wire(plant, 'Actual physical state/1', 'State s/1');

names = {'pdot','qdot','vdot','rdot','a1dot','a2dot'};
expressions = {'u(3)', 'u(4)', ...
    '(u(5)+u(6))/2-cap_dv*u(3)', ...
    '(u(5)-u(6))/2-cap_k*sin(u(2))-cap_dr*u(4)+u(9)', ...
    '(-u(5)+u(7))/cap_tau1', '(-u(6)+u(8))/cap_tau2'};
for index = 1:numel(names)
    y = 55 + (index - 1) * 65;
    add_block('simulink/User-Defined Functions/Fcn', [plant '/' names{index}], ...
        'Expr', expressions{index}, 'Position', [270 y 580 y + 35]);
    wire(plant, 'State input disturbance/1', [names{index} '/1']);
    wire(plant, [names{index} '/1'], ['State derivatives/' num2str(index)]);
end
annotation = Simulink.Annotation(plant, ...
    ['s = [p; q; v; r; a1; a2]. u is already clipped by the model-level Saturation block.' newline ...
    'Both actuator forces affect translation; their difference affects the internal angle.' newline ...
    'Six native Fcn derivatives feed one six-state continuous Integrator.']);
annotation.Position = [60 460];
end

function wire(system, source, destination)
add_line(system, source, destination, 'autorouting', 'on');
end
