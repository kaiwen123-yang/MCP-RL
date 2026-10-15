function report = runCodesignMathAudit(outputDir)
% RUNCODESIGNMATHAUDIT Check realizability, coordinate equivalence and feedback response.
% This is an analytical/numerical audit, not robot simulation or RL training.
% Files written: a JSON result and a MAT result in outputDir.

arguments
    outputDir (1, 1) string
end

if ~isfolder(outputDir)
    mkdir(outputDir)
end

symbolicResult = checkSymbolicRelations();
boxResult = checkBoxRealizability();
equivalenceResult = checkMpcEquivalence();
causalResult = checkCausalFeedback();

report = struct( ...
    scope="Mathematical audit only; no learning or robot performance claims", ...
    matlabRelease=string(version("-release")), ...
    symbolic=symbolicResult, ...
    boxRealizability=boxResult, ...
    coordinateEquivalence=equivalenceResult, ...
    causalFeedback=causalResult);

resultText = jsonencode(report, PrettyPrint=true);
fileId = fopen(fullfile(outputDir, "math-audit-results.json"), "w", "n", "UTF-8");
assert(fileId >= 0, "Cannot open the requested result file.");
fileCleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, "%s\n", resultText);
save(fullfile(outputDir, "math-audit-results.mat"), "report");
disp(resultText)
end

function result = checkSymbolicRelations()
syms plantA plantB refA refB feedbackGain z e command disturbance real
control = ((refA-plantA)*z + refB*command - feedbackGain*e)/plantB;
actualNext = plantA*(z+e) + plantB*control + disturbance;
referenceNext = refA*z + refB*command;
errorResidual = simplify(actualNext-referenceNext-((plantA-feedbackGain)*e+disturbance));

syms actualInput real
mappedCommand = ((plantA-refA)*z + plantB*actualInput)/refB;
coordinateResidual = simplify(refA*z+refB*mappedCommand-(plantA*z+plantB*actualInput));

syms slowCommand gain noise0 noise1 target real
firstState = slowCommand + noise0;
secondInput = slowCommand - gain*firstState;
secondState = firstState + secondInput + noise1;
twoStepResidual = simplify(secondState-((2-gain)*slowCommand+(1-gain)*noise0+noise1));
matchedError = simplify(subs(secondState, slowCommand, target/(2-gain))-target);

assert(isAlways(errorResidual == 0), "Error recursion does not simplify to zero.");
assert(isAlways(coordinateResidual == 0), "Command reparameterization identity failed.");
assert(isAlways(twoStepResidual == 0), "Two-step causal-response identity failed.");
result = struct( ...
    errorRecursionVerified=true, ...
    invertibleCommandMapVerified=true, ...
    twoStepResponseVerified=true, ...
    matchedEndpointError=string(matchedError), ...
    assumptions="plantB and refB nonzero; gain differs from 2 in endpoint matching");
end

function result = checkBoxRealizability()
plantA = 1.05;
plantB = 0.8;
stateRadius = 1;
commandRadius = 0.6;
errorRadius = 0.2;
noiseRadius = 0.04;
inputLimit = 1;
parameters = [0.7, 0.4, 0.5; 0.2, 1.2, 0.5; 0.7, 0.4, 0.1];
labels = ["feasible", "input_violation", "error_invariance_violation"];
[zSign, cSign, eSign, wSign] = ndgrid([-1, 1], [-1, 1], [-1, 1], [-1, 1]);
stateVertex = stateRadius*zSign(:);
commandVertex = commandRadius*cSign(:);
errorVertex = errorRadius*eSign(:);
noiseVertex = noiseRadius*wSign(:);
caseCells = cell(size(parameters, 1), 1);
for caseIndex = 1:size(parameters, 1)
    refA = parameters(caseIndex, 1);
    refB = parameters(caseIndex, 2);
    feedback = parameters(caseIndex, 3);
    inputVertex = ((refA-plantA)*stateVertex + refB*commandVertex - feedback*errorVertex)/plantB;
    errorNext = (plantA-feedback)*errorVertex + noiseVertex;
    referenceNext = refA*stateVertex + refB*commandVertex;
    analyticInput = (abs(refA-plantA)*stateRadius + abs(refB)*commandRadius ...
        + abs(feedback)*errorRadius)/abs(plantB);
    analyticError = abs(plantA-feedback)*errorRadius + noiseRadius;
    analyticReference = abs(refA)*stateRadius + abs(refB)*commandRadius;
    vertexResidual = max(abs([max(abs(inputVertex))-analyticInput, ...
        max(abs(errorNext))-analyticError, max(abs(referenceNext))-analyticReference]));
    assert(vertexResidual < 1e-12, "Vertex and support-function bounds disagree.");
    caseCells{caseIndex} = struct( ...
        label=labels(caseIndex), ...
        parameters=parameters(caseIndex, :), ...
        maximumInput=analyticInput, ...
        maximumNextError=analyticError, ...
        maximumReferenceState=analyticReference, ...
        inputFeasible=analyticInput <= inputLimit+1e-12, ...
        errorInvariant=analyticError <= errorRadius+1e-12, ...
        referenceInvariant=analyticReference <= stateRadius+1e-12, ...
        vertexFormulaResidual=vertexResidual);
end

cases = vertcat(caseCells{:});
assert(cases(1).inputFeasible && cases(1).errorInvariant && cases(1).referenceInvariant, ...
    "The intended feasible example failed.");
assert(~cases(2).inputFeasible, "The intended input counterexample was not detected.");
assert(~cases(3).errorInvariant, "The intended error counterexample was not detected.");
result = struct(parametersOrder=["referenceA", "referenceB", "feedback"], ...
    plantA=plantA, plantB=plantB, stateRadius=stateRadius, ...
    commandRadius=commandRadius, errorRadius=errorRadius, ...
    noiseRadius=noiseRadius, inputLimit=inputLimit, cases=cases);
end

function result = checkMpcEquivalence()
plantA = 1.08;
plantB = 0.18;
horizon = 18;
initialState = -0.6;
inputLimit = 1;
stateLimit = 2.5;
targetState = ones(horizon, 1);
stateWeights = diag([ones(1, horizon-1), 6]);
inputWeights = 0.06*eye(horizon);
options = optimoptions("quadprog", Display="off", ...
    OptimalityTolerance=1e-10, ConstraintTolerance=1e-10, MaxIterations=1000);
[plantFree, plantResponse] = predictionMatrices(plantA, plantB, horizon);
plantOffset = plantFree*initialState;
physicalHessian = 2*(plantResponse'*stateWeights*plantResponse + inputWeights);
physicalHessian = (physicalHessian+physicalHessian')/2;
physicalGradient = 2*plantResponse'*stateWeights*(plantOffset-targetState);
[physicalInput, ~, exitFlag] = quadprog(physicalHessian, physicalGradient, ...
    [plantResponse; -plantResponse], ...
    [stateLimit-plantOffset; stateLimit+plantOffset], [], [], ...
    -inputLimit*ones(horizon, 1), inputLimit*ones(horizon, 1), [], options);
assert(exitFlag > 0, "The physical MPC QP did not converge.");
physicalState = plantOffset + plantResponse*physicalInput;
physicalCost = taskCost(physicalState, physicalInput, targetState, stateWeights, inputWeights);

referenceParameters = [0.4, 0.4; 0.4, 0.9; 0.8, 0.4; 0.8, 0.9; 1.1, 0.4; 1.1, 0.9];
caseCells = cell(size(referenceParameters, 1), 1);
for caseIndex = 1:size(referenceParameters, 1)
    refA = referenceParameters(caseIndex, 1);
    refB = referenceParameters(caseIndex, 2);
    [referenceFree, referenceResponse] = predictionMatrices(refA, refB, horizon);
    referenceOffset = referenceFree*initialState;
    previousFree = [1; referenceFree(1:end-1)]*initialState;
    previousResponse = [zeros(1, horizon); referenceResponse(1:end-1, :)];
    inputMap = ((refA-plantA)/plantB)*previousResponse + (refB/plantB)*eye(horizon);
    inputOffset = ((refA-plantA)/plantB)*previousFree;
    hessian = 2*(referenceResponse'*stateWeights*referenceResponse + inputMap'*inputWeights*inputMap);
    gradient = 2*(referenceResponse'*stateWeights*(referenceOffset-targetState) ...
        + inputMap'*inputWeights*inputOffset);
    inequalities = [inputMap; -inputMap; referenceResponse; -referenceResponse];
    limits = [inputLimit-inputOffset; inputLimit+inputOffset; ...
        stateLimit-referenceOffset; stateLimit+referenceOffset];
    [command, ~, transformedExit] = quadprog((hessian+hessian')/2, gradient, ...
        inequalities, limits, [], [], [], [], [], options);
    assert(transformedExit > 0, "A transformed MPC QP did not converge.");
    actualInput = inputOffset + inputMap*command;
    actualState = plantOffset + plantResponse*actualInput;
    virtualState = referenceOffset + referenceResponse*command;
    actualCost = taskCost(actualState, actualInput, targetState, stateWeights, inputWeights);
    caseCells{caseIndex} = struct(referenceA=refA, referenceB=refB, ...
        cost=actualCost, costGap=actualCost-physicalCost, ...
        inputDifference=max(abs(actualInput-physicalInput)), ...
        stateDifference=max(abs(actualState-physicalState)), ...
        virtualPhysicalMismatch=max(abs(actualState-virtualState)), ...
        solverExit=transformedExit);
end
cases = vertcat(caseCells{:});
maximumInputDifference = max([cases.inputDifference]);
maximumCostGap = max(abs([cases.costGap]));
assert(maximumInputDifference < 1e-5, "Equivalent MPCs returned materially different physical inputs.");
assert(maximumCostGap < 1e-7, "Equivalent MPCs returned materially different physical costs.");
result = struct(horizon=horizon, physicalCost=physicalCost, ...
    maximumInputDifference=maximumInputDifference, maximumAbsoluteCostGap=maximumCostGap, ...
    cases=cases, conclusion="Changing an invertible command coordinate gives no added control capability.");
end

function result = checkCausalFeedback()
noiseRadius = 0.1;
inputLimit = 1;
target = 1.5;
terminalTolerance = 0.15;
gains = [0, 0.5, 1];
[firstNoise, secondNoise] = ndgrid([-noiseRadius, noiseRadius], [-noiseRadius, noiseRadius]);
caseCells = cell(numel(gains), 1);
for gainIndex = 1:numel(gains)
    gain = gains(gainIndex);
    command = target/(2-gain);
    firstState = command + firstNoise;
    secondInput = command - gain*firstState;
    endpoint = firstState + secondInput + secondNoise;
    endpointError = max(abs(endpoint-target), [], "all");
    inputMaximum = max(abs([command; secondInput(:)]));
    predictedError = (2-gain)*noiseRadius;
    predictedInput = max(command, (1-gain)*command+gain*noiseRadius);
    assert(abs(endpointError-predictedError) < 1e-12, "Causal error support formula failed.");
    assert(abs(inputMaximum-predictedInput) < 1e-12, "Causal input support formula failed.");
    caseCells{gainIndex} = struct(gain=gain, command=command, ...
        nominalEndpoint=target, worstEndpointError=endpointError, maximumInput=inputMaximum, ...
        inputFeasible=inputMaximum <= inputLimit+1e-12, ...
        terminalFeasible=endpointError <= terminalTolerance+1e-12);
end

cases = vertcat(caseCells{:});
% Exact task-specific classical design on gain in [0,1], with U >= disturbance radius.
minimumGain = max(0, 2-terminalTolerance/noiseRadius);
optimalGain = minimumGain;
maximumTarget = (2-optimalGain)*inputLimit;
assert(optimalGain <= 1, "The specified tolerance is infeasible in this feedback class.");
assert(cases(2).inputFeasible && cases(2).terminalFeasible, "The balanced response should be feasible.");
assert(~cases(1).terminalFeasible && ~cases(3).inputFeasible, "The intended two tradeoffs were not found.");
result = struct( ...
    model="x(j+1)=x(j)+u(j)+w(j); u(j)=command-gain*x(j); x(0)=0; two fast steps", ...
    noiseRadius=noiseRadius, inputLimit=inputLimit, target=target, terminalTolerance=terminalTolerance, ...
    cases=cases, optimalClassicalGain=optimalGain, maximumClassicalTarget=maximumTarget, ...
    claimedAdvantageOverOptimallyTunedClassicalControl=0, ...
    conclusion="Fast feedback changes disturbance response after matching nominal endpoints; this alone is not an RL innovation.");
end

function [freeResponse, inputResponse] = predictionMatrices(stateGain, inputGain, horizon)
freeResponse = stateGain.^(1:horizon)';
inputResponse = zeros(horizon);
for rowIndex = 1:horizon
    for columnIndex = 1:rowIndex
        inputResponse(rowIndex, columnIndex) = inputGain*stateGain^(rowIndex-columnIndex);
    end
end
end

function value = taskCost(state, input, target, stateWeights, inputWeights)
error = state - target;
value = error'*stateWeights*error + input'*inputWeights*input;
end
