function result = runLearningPilotReplay(dataFile, outputDirectory)
%RUNLEARNINGPILOTREPLAY Verify a frozen Python actor in MATLAB and Simulink.
%   Expected export fields: W1,b1,W2,b2,W3,b3, command[N,1],
%   disturbance[N,1], initial_state[3,1], state_expected[N,3],
%   action_expected[N,1]. State_expected rows are post-action states.
%   This is numerical low-level feedback verification under recorded MPC
%   commands, not a second MPC implementation or robot validation.
arguments
    dataFile (1,1) string
    outputDirectory (1,1) string = ""
end
sourceDirectory = string(fileparts(mfilename('fullpath')));
addpath(sourceDirectory);
if strlength(outputDirectory) == 0
    outputDirectory = fullfile(fileparts(dataFile), 'matlab-simulink');
end
if ~isfolder(outputDirectory)
    mkdir(outputDirectory);
end
data = load(dataFile);
required = {'W1','b1','W2','b2','W3','b3','command','disturbance', ...
    'initial_state','state_expected','action_expected'};
assert(all(isfield(data, required)), 'Export is missing a required field.');
net = struct('W1', single(data.W1), 'b1', single(data.b1(:)), ...
    'W2', single(data.W2), 'b2', single(data.b2(:)), ...
    'W3', single(data.W3), 'b3', single(data.b3(:)));
assert(isequal(size(net.W1), [32 12]) && isequal(size(net.W2), [32 32]) ...
    && isequal(size(net.W3), [1 32]), 'Expected a 12-32-32-1 actor.');
cfg = struct('dt', 0.02, 'tau', 0.12, 'k', 1, 'd', 0.2, ...
    'drag', 0.1, 'U', 1.2, 'W', 0.04);
initialState = single(data.initial_state(:));
command = single(data.command(:));
disturbance = single(data.disturbance(:));
nSteps = numel(command);
assert(numel(initialState) == 3 && numel(disturbance) == nSteps ...
    && isequal(size(data.state_expected), [nSteps 3]) ...
    && numel(data.action_expected) == nSteps, 'Export trace dimensions disagree.');
matlabState = zeros(nSteps, 3, 'single');
matlabAction = zeros(nSteps, 1, 'single');
state = initialState;
for index = 1:nSteps
    action = learningPilotActor(state, command(index), net);
    state = learningPilotPlantStep(state, action, disturbance(index), cfg);
    matlabAction(index) = action;
    matlabState(index, :) = state';
end

modelFile = buildLearningPilotModel(outputDirectory, net, cfg, initialState, command, disturbance);
[~, modelName] = fileparts(modelFile);
simulation = sim(modelName, 'SimulationMode', 'normal');
simulinkState = single(simulation.get('stateLog'));
simulinkAction = single(simulation.get('actionLog'));
simulinkAction = simulinkAction(:);
assert(isequal(size(simulinkState), [nSteps 3]) ...
    && numel(simulinkAction) == nSteps, 'Unexpected Simulink logging/sample count.');
stateExpected = single(data.state_expected);
actionExpected = single(data.action_expected(:));
result = struct;
result.scope = ['Frozen full-action actor plus actual nonlinear/actuator feedback; ' ...
    'recorded MPC commands, no high-level planner re-execution'];
result.input_file = char(dataFile);
result.model_file = char(modelFile);
result.samples = nSteps;
result.dt_seconds = cfg.dt;
result.rk4_substeps = 2;
result.precision = 'single';
result.physical = cfg;
result.actor_parameter_count = numel(net.W1)+numel(net.b1)+numel(net.W2) ...
    +numel(net.b2)+numel(net.W3)+numel(net.b3);
result.actions_recomputed_from_simulated_state = true;
result.high_level_mpc_rerun = false;
if isfield(data, 'response_raw')
    result.realization_trace_check = learningPilotRealizationCheck(initialState, ...
        matlabState, matlabAction, command, data.response_raw, cfg);
end
result.matlab_python_state_max_error = max(abs(double(matlabState)-double(stateExpected)), [], 'all');
result.matlab_python_action_max_error = max(abs(double(matlabAction)-double(actionExpected)), [], 'all');
result.simulink_python_state_max_error = max(abs(double(simulinkState)-double(stateExpected)), [], 'all');
result.simulink_python_action_max_error = max(abs(double(simulinkAction)-double(actionExpected)), [], 'all');
result.simulink_matlab_state_max_error = max(abs(double(simulinkState)-double(matlabState)), [], 'all');
result.simulink_matlab_action_max_error = max(abs(double(simulinkAction)-double(matlabAction)), [], 'all');
result.sample_time_max_error = max(abs(double(simulation.tout(:))-(0:nSteps-1)'*cfg.dt));
result.numeric_tolerance = 5e-5;
result.passed = max([result.matlab_python_state_max_error, ...
    result.matlab_python_action_max_error, result.simulink_python_state_max_error, ...
    result.simulink_python_action_max_error, result.simulink_matlab_state_max_error, ...
    result.simulink_matlab_action_max_error]) <= result.numeric_tolerance ...
    && result.sample_time_max_error < 1e-10;
save(fullfile(outputDirectory, 'feedback-check.mat'), 'matlabState', 'matlabAction', ...
    'simulinkState', 'simulinkAction', 'stateExpected', 'actionExpected', 'result');
file = fopen(fullfile(outputDirectory, 'feedback-check.json'), 'w');
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', jsonencode(result, 'PrettyPrint', true));
disp(jsonencode(result, 'PrettyPrint', true));
assert(result.passed, 'Python/MATLAB/Simulink feedback replay exceeds numerical tolerance.');
end
