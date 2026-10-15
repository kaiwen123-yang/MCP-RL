function evidence = summarizeNonlinearEvidence(outputDir)
% SUMMARIZENONLINEAREVIDENCE Separate realization, task, and optimization evidence.
arguments
    outputDir (1, 1) string
end
loaded = load(fullfile(outputDir, "nonlinear-trajectories.mat"));
p = loaded.report.bestParameters;
certificate = loaded.report.certificate;
P = certificate.P;
inverseP = P\eye(3);
actuatorRows = [-p.kp-p.k, p.d-p.kv, 1; -p.kp+p.k, p.d-p.kv, 1];
actuatorGain = max(sqrt(sum((actuatorRows*inverseP).*actuatorRows, 2)));
errorBounds = [certificate.errorCoordinateBounds(1:2); ...
    certificate.radius*actuatorGain; certificate.inputErrorBound+certificate.holdBound];
[gainBounds, gainCertificates] = certifyEnergyGains(certificate.A, certificate.B, [0; 0; 1/p.tau]);
errorEnergy = gainBounds(:, 1)*p.W*sqrt(p.tStop) ...
    +gainBounds(:, 2)*certificate.holdBound*sqrt(p.tStop);
ae = -p.kp*(1+p.tau*(p.d-p.kv));
by = p.d-p.kv-p.tau*((p.d-p.kv)*p.kv+p.kp+p.rho);
ce = 1+p.tau*(p.d-p.kv-p.ka);
actuatorEnergy = [p.k+p.kp, abs(p.d-p.kv), 1]*errorEnergy;
inputEnergy = [abs(ae)+p.k*(1+p.tau*p.sBound), abs(by)+p.tau*p.k, abs(ce)]*errorEnergy ...
    +certificate.holdBound*sqrt(p.tStop);
physicalErrorEnergy = [errorEnergy(1:2); actuatorEnergy; inputEnergy];
trajectories = {loaded.designOne, loaded.designTwo, loaded.heldout};
taskRecords = cell(3, 1);
for index = 1:3
    trajectory = trajectories{index};
    z = trajectory.state;
    q = -p.omega^2*(z(:, 5)-z(:, 4))-2*p.zeta*p.omega*z(:, 6);
    j = -p.omega^2*(z(:, 6)-trajectory.command)-2*p.zeta*p.omega*q;
    nominalActuator = p.k*sin(z(:, 5))+p.d*z(:, 6)+q;
    nominalInput = nominalActuator+p.tau*(p.k*cos(z(:, 5)).*z(:, 6)+p.d*q+j);
    referenceCostRate = (z(:, 5)-trajectory.target).^2+0.05*z(:, 6).^2 ...
        +0.01*nominalActuator.^2+0.005*nominalInput.^2;
    referenceCost = trapz(trajectory.time, referenceCostRate);
    if index == 1
        targetBound = 0.24;
        targetDerivative = 0.24*2*pi/9;
    elseif index == 2
        targetBound = 0.23;
        targetDerivative = 0;
    else
        targetBound = 0.23;
        targetDerivative = 0.17*2*pi/7+0.06*2*pi/3.7;
    end
    nominalBounds = [p.rBound+targetBound; p.sBound; ...
        p.k*sin(p.rBound)+p.d*p.sBound+p.qBound; certificate.inputBudget];
    weights = [1; 0.05; 0.01; 0.005];
    uniformCostGapBound = p.tStop*sum(weights.*errorBounds.*(2*nominalBounds+errorBounds));
    nominalSignals = [z(:, 5)-trajectory.target, z(:, 6), nominalActuator, nominalInput];
    inputDerivative = p.k*p.sBound+p.d*p.qBound+p.jBound ...
        +p.tau*(p.k*(p.sBound^2+p.qBound)+p.d*p.jBound ...
        +p.omega^2*p.qBound+2*p.zeta*p.omega*p.jBound);
    signalDerivatives = [p.sBound+targetDerivative, p.qBound, ...
        p.k*p.sBound+p.d*p.qBound+p.jBound, inputDerivative];
    leftSignals = nominalSignals(1:end-1, :);
    nominalEnergyUpper = sqrt(sum(p.h*leftSignals.^2 ...
        +p.h^2*abs(leftSignals).*signalDerivatives ...
        +(p.h^3/3)*signalDerivatives.^2, 1))';
    energyCostGapBound = sum(weights.*physicalErrorEnergy.*(2*nominalEnergyUpper+physicalErrorEnergy));
    taskRecords{index} = struct(taskId=index, actualCost=trajectory.cost, ...
        referenceCostNumerical=referenceCost, actualReferenceCostGap=abs(trajectory.cost-referenceCost), ...
        uniformCostGapBound=uniformCostGapBound, ...
        energyCostGapBound=energyCostGapBound, ...
        boundExceedsActualCost=uniformCostGapBound > trajectory.cost, ...
        rmsRealizationError=sqrt(mean((z(:, 1)-z(:, 5)).^2)), ...
        maximumRealizationError=max(abs(z(:, 1)-z(:, 5))), ...
        taskRmsError=trajectory.rmsTaskError, maximumInput=trajectory.maxRawInput);
end

% Independently differentiate the Lyapunov equation with respect to kv.
A = certificate.A;
dA = [0, 0, 0; 0, -1, 0; 0, 0, 0];
dP = lyap(A', dA'*P+P*dA);
step = 1e-5;
plusP = lyap((A+step*dA)', eye(3));
minusP = lyap((A-step*dA)', eye(3));
derivativeError = norm(dP-(plusP-minusP)/(2*step), "fro");
assert(derivativeError < 1e-7, "Lyapunov sensitivity and finite difference disagree.");

% Same-dimensional feasible-update example. Mature SQP is an explicit comparator.
objective = @(x) sum((x-1).^2);
options = optimoptions("fmincon", Algorithm="sqp", Display="off", ...
    OptimalityTolerance=1e-10, ConstraintTolerance=1e-10);
[solution, objectiveValue, exitFlag] = fmincon(objective, [0; 0], [], [], ...
    [-1, 1], 0, [-2; -2], [2; 2], [], options);
assert(exitFlag > 0 && objectiveValue < 1e-12, "Classical SQP failed the coupled-update check.");
updateCheck = struct(start=[0, 0], initialCost=2, exactFeasibleBlockUpdateCost=2, ...
    jointSqpSolution=solution, jointSqpCost=objectiveValue, ...
    interpretation="Block-wise strict feasibility can stall; standard classical SQP already solves this example.");
evidence = struct(tasks=vertcat(taskRecords{:}), errorBounds=errorBounds, ...
    energyGainBounds=gainBounds, gainCertificates=vertcat(gainCertificates{:}), ...
    physicalErrorEnergyBounds=physicalErrorEnergy, ...
    lyapunovSensitivityError=derivativeError, feasibleUpdateCheck=updateCheck, ...
    performanceBoundScope="Uniform analytical gap under the proved domain; numerical reference integral is diagnostic", ...
    limitation="No new RL superiority; robust realization and useful task improvement are distinct claims");
fileId = fopen(fullfile(outputDir, "evidence-summary.json"), "w", "n", "UTF-8");
assert(fileId >= 0, "Cannot create the evidence output.");
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, "%s\n", jsonencode(evidence, PrettyPrint=true));
disp(jsonencode(evidence, PrettyPrint=true));
end

function [bounds, certificates] = certifyEnergyGains(A, B, sampleChannel)
channels = [B, sampleChannel];
bounds = zeros(3, 2);
certificates = cell(6, 1);
index = 0;
for output = 1:3
    outputRow = zeros(1, 3);
    outputRow(output) = 1;
    for input = 1:2
        channel = channels(:, input);
        estimatedNorm = norm(ss(A, channel, outputRow, 0), Inf);
        gamma = ceil(1.02*estimatedNorm*1e10)/1e10;
        setlmis([]);
        variable = lmivar(1, [3, 1]);
        lmiterm([1, 1, 1, variable], A', 1, 's');
        lmiterm([1, 1, 1, 0], outputRow'*outputRow);
        lmiterm([1, 1, 2, variable], 1, channel);
        lmiterm([1, 2, 2, 0], -gamma^2);
        lmiterm([2, 1, 1, variable], -1, 1);
        system = getlmis;
        [feasibility, solution] = feasp(system, [1e-7, 200, 1e6, 0, 1]);
        assert(feasibility < 0, "A proposed energy gain has no bounded-real certificate.");
        X = dec2mat(system, solution, variable);
        X = round((X+X')/2, 10);
        lmi = [A'*X+X*A+outputRow'*outputRow, X*channel; channel'*X, -gamma^2];
        assert(min(eig(X)) > 0 && max(eig(lmi)) < 0, "Rounded gain certificate lost strict feasibility.");
        bounds(output, input) = gamma;
        index = index+1;
        certificates{index} = struct(outputIndex=output, inputIndex=input, ...
            estimatedNorm=estimatedNorm, gammaDecimal=compose("%.10f", gamma), ...
            XDecimal=compose("%.10f", X), ADecimal=compose("%.10f", A), ...
            BDecimal=compose("%.10f", channel), CDecimal=compose("%.10f", outputRow), ...
            smallestXEigenvalue=min(eig(X)), largestLmiEigenvalue=max(eig(lmi)));
    end
end
end
