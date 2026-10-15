function report = runNonlinearPrototype(outputDir, reuseSearch, performLocalSqp)
% RUNNONLINEARPROTOTYPE Constrained nonlinear co-design and matched classical audit.
% No neural network training is performed. Searches an explicitly finite class.
% The same evaluated candidates are used for joint and nested fixed-response design.

arguments
    outputDir (1, 1) string
    reuseSearch (1, 1) logical = false
    performLocalSqp (1, 1) logical = false
end
if ~isfolder(outputDir)
    mkdir(outputDir)
end
timer = tic;
base = struct(k=1, d=0.2, tau=0.1, U=0.95, W=0.02, h=0.001, ...
    disturbanceStep=0.001, highStep=0.1, horizon=10, tStop=8, x0=zeros(6, 1), ...
    omega=1.2, zeta=1, kp=4, kv=3, ka=8, rho=1, ...
    rBound=0.35, sBound=0.25, qBound=0.3, jBound=0.5);
[om, ze, kp, kv, ka] = ndgrid([0.8, 1.4], [0.8, 1.2], [3, 5], [2.5, 3.5], [7, 10]);
candidateValues = [om(:), ze(:), kp(:), kv(:), ka(:), ones(numel(om), 1)];
[om, ze, kp, kv, ka, rho] = ndgrid([0.6, 1, 1.4], [0.7, 1, 1.3], ...
    [2, 4, 6], [2, 3, 4], [5, 8, 11], [0.5, 1.5]);
candidateValues = [candidateValues; om(:), ze(:), kp(:), kv(:), ka(:), rho(:)];
candidateValues = [candidateValues; 1.2, 1, 4, 3, 8, 1];
candidateValues = unique(candidateValues, "rows", "stable");
parameterNames = ["omega", "zeta", "kp", "kv", "ka", "rho"];
candidateCount = size(candidateValues, 1);
score = inf(candidateCount, 1);
candidateRecords = cell(candidateCount, 1);
searchPath = fullfile(outputDir, "finite-search.mat");
searchDefinition = "terminal-polytope-active-set-grid519-v2";
if reuseSearch
    previous = load(searchPath, "base", "candidateValues", "score", "candidateRecords", "searchDefinition");
    assert(isequal(previous.base, base) && isequal(previous.candidateValues, candidateValues), ...
        "Saved candidate search does not match this experiment definition.");
    assert(previous.searchDefinition == searchDefinition, "The saved planner definition is different.");
    score = previous.score;
    candidateRecords = previous.candidateRecords;
else
for candidateIndex = 1:candidateCount
    params = setCandidate(base, candidateValues(candidateIndex, :));
    certificate = realizationCertificate(params);
    record = struct(index=candidateIndex, parameters=candidateValues(candidateIndex, :), ...
        certificateFeasible=certificate.feasible, radius=certificate.radius, ...
        sampleHoldBound=certificate.holdBound, referenceInputBudget=certificate.inputBudget, ...
        meanDesignCost=[], qpSuccess=false, planningFeasible=false);
    if certificate.feasible
        planner = makePlanner(params, certificate);
        first = simulateCase(params, certificate, planner, 1, 1, false);
        second = simulateCase(params, certificate, planner, 2, 1, false);
        record.qpSuccess = first.qpSuccess && second.qpSuccess;
        record.planningFeasible = first.planningFeasible && second.planningFeasible;
        if record.planningFeasible
            score(candidateIndex) = mean([first.cost, second.cost]);
            record.meanDesignCost = score(candidateIndex);
        end
    end
    candidateRecords{candidateIndex} = record;
    if mod(candidateIndex, 8) == 0 || candidateIndex == candidateCount
        fprintf("Evaluated %d/%d declared designs.\n", candidateIndex, candidateCount);
    end
end
save(searchPath, "base", "candidateValues", "score", "candidateRecords", "searchDefinition");
end
assert(any(isfinite(score)), "The declared candidate class contains no feasible solved design.");
[bestScore, bestIndex] = min(score);
bestParams = setCandidate(base, candidateValues(bestIndex, :));
bestCertificate = realizationCertificate(bestParams);
bestPlanner = makePlanner(bestParams, bestCertificate);
localSqpStudy = struct(executed=false);
if performLocalSqp
    localSqpStudy = runLocalSqp(base, candidateValues(bestIndex, :), bestScore);
end

% Nested search: fully tune feedback for each fixed response, then select response.
responsePairs = unique(candidateValues(:, 1:2), "rows", "stable");
nestedScores = inf(size(responsePairs, 1), 1);
nestedIndices = zeros(size(responsePairs, 1), 1);
for responseIndex = 1:size(responsePairs, 1)
    mask = all(abs(candidateValues(:, 1:2)-responsePairs(responseIndex, :)) < 1e-12, 2);
    indices = find(mask);
    [nestedScores(responseIndex), localIndex] = min(score(indices));
    nestedIndices(responseIndex) = indices(localIndex);
end
[bestNestedScore, nestedRow] = min(nestedScores);
bestNestedIndex = nestedIndices(nestedRow);
assert(abs(bestNestedScore-bestScore) < 1e-12, "Matched finite searches should have identical optimum.");

designOne = simulateCase(bestParams, bestCertificate, bestPlanner, 1, 1, true);
designTwo = simulateCase(bestParams, bestCertificate, bestPlanner, 2, 1, true);
heldout = simulateCase(bestParams, bestCertificate, bestPlanner, 3, 1, true);
fineIntegration = simulateCase(bestParams, bestCertificate, bestPlanner, 3, 2, true);
halfParams = bestParams;
halfParams.h = bestParams.h/2;
halfCertificate = realizationCertificate(halfParams);
halfPlanner = makePlanner(halfParams, halfCertificate);
% Keep the original planning constraints to isolate the sampling change.
halfPlanner = copyPlannerConstraints(halfPlanner, bestPlanner);
halfSampling = simulateCase(halfParams, halfCertificate, halfPlanner, 3, 1, true);
halfAligned = interp1(halfSampling.time, halfSampling.state, heldout.time, "linear");
validation = struct( ...
    independentControlFormulaError=heldout.controlFormulaError, ...
    halfIntegrationStepStateError=max(abs(fineIntegration.state-heldout.state), [], "all"), ...
    halfControllerPeriodStateError=max(abs(halfAligned-heldout.state), [], "all"), ...
    designOne=stripTrajectory(designOne), designTwo=stripTrajectory(designTwo), ...
    heldout=stripTrajectory(heldout));

simulinkResult = verifySimulink(outputDir, bestParams, heldout);
report = struct( ...
    scope="Nonlinear analytical/classical prototype; no RL training and no new-method superiority claim", ...
    parameterOrder=parameterNames, baseParameters=base, ...
    candidateCount=candidateCount, certifiedCandidateCount=sum(cellfun(@(r) r.certificateFeasible, candidateRecords)), ...
    solvedCandidateCount=sum(isfinite(score)), candidates=vertcat(candidateRecords{:}), ...
    bestJointIndex=bestIndex, bestNestedFixedIndex=bestNestedIndex, ...
    jointDesignCost=bestScore, bestFixedResponseCost=bestNestedScore, ...
    matchedClassAdvantage=bestNestedScore-bestScore, ...
    bestParameters=bestParams, certificate=bestCertificate, terminalSet=bestPlanner.terminalInfo, ...
    localSqpStudy=localSqpStudy, ...
    validation=validation, simulink=simulinkResult, elapsedSeconds=toc(timer), ...
    searchReused=reuseSearch, ...
    comparisonMode="Same physical evaluation table; joint and nested minima are an equivalence check", ...
    qualification="Exhaustive optimum only within the declared finite class; no global continuous tuning claim");
writeJson(fullfile(outputDir, "nonlinear-results.json"), report);
save(fullfile(outputDir, "nonlinear-trajectories.mat"), "report", "designOne", "designTwo", "heldout");
renderResults(outputDir, report, heldout);
disp(jsonencode(struct(candidateCount=report.candidateCount, ...
    certified=report.certifiedCandidateCount, solved=report.solvedCandidateCount, ...
    bestParameters=candidateValues(bestIndex, :), cost=bestScore, ...
    matchedClassAdvantage=report.matchedClassAdvantage, certificate=bestCertificate, ...
    validation=validation, simulink=simulinkResult, localSqpStudy=localSqpStudy, ...
    elapsedSeconds=report.elapsedSeconds), PrettyPrint=true));
end

function study = runLocalSqp(base, initial, initialCost)
callCount = 0;
bestFound = initial;
bestCost = initialCost;
options = optimoptions("fmincon", Algorithm="sqp", Display="off", ...
    MaxFunctionEvaluations=70, MaxIterations=8, FiniteDifferenceStepSize=1e-3, ...
    OptimalityTolerance=1e-4, StepTolerance=1e-5);
lower = [0.6, 0.7, 2, 2, 5, 0.5];
upper = [1.4, 1.3, 6, 4, 11, 1.5];
[solution, objective, exitFlag, solverOutput] = fmincon(@evaluate, initial, ...
    [], [], [], [], lower, upper, [], options);
study = struct(executed=true, method="Classical SQP on the same six physical design parameters", ...
    initialParameters=initial, initialCost=initialCost, solution=solution, ...
    returnedCost=objective, bestFeasibleParameters=bestFound, bestFeasibleCost=bestCost, ...
    exitFlag=exitFlag, evaluationCount=callCount, iterations=solverOutput.iterations, ...
    limitation="Feasibility-gated numerical objective; local finite-budget result, not a global convergence theorem");

    function value = evaluate(values)
        callCount = callCount+1;
        params = setCandidate(base, values);
        cert = realizationCertificate(params);
        if ~cert.feasible
            value = 1e3;
            return
        end
        planner = makePlanner(params, cert);
        one = simulateCase(params, cert, planner, 1, 1, false);
        two = simulateCase(params, cert, planner, 2, 1, false);
        if ~(one.planningFeasible && two.planningFeasible)
            value = 1e3;
            return
        end
        value = mean([one.cost, two.cost]);
        if value < bestCost
            bestCost = value;
            bestFound = values;
        end
    end
end

function params = setCandidate(base, values)
params = base;
params.omega = values(1);
params.zeta = values(2);
params.kp = values(3);
params.kv = values(4);
params.ka = values(5);
params.rho = values(6);
end

function cert = realizationCertificate(p)
A = [0, 1, 0; -p.kp, -p.kv, 1; 0, -p.rho, -p.ka];
B = [0; 1; p.kv-p.d];
C = [0; 0; 1/p.tau];
P = lyap(A', eye(3));
inverseP = P\eye(3);
alpha = 1/(2*max(eig(P)));
beta = sqrt(B'*P*B);
betaSample = sqrt(C'*P*C);
ae = -p.kp*(1+p.tau*(p.d-p.kv));
by = p.d-p.kv-p.tau*((p.d-p.kv)*p.kv+p.kp+p.rho);
ce = 1+p.tau*(p.d-p.kv-p.ka);
[signOne, signTwo] = ndgrid([-1, 1], [-1, 1]);
rows = [ae+p.k*(1+p.tau*p.sBound)*signOne(:), ...
    by+p.tau*p.k*signTwo(:), ce*ones(4, 1)];
inputGain = max(sqrt(sum((rows*inverseP).*rows, 2)));
radiusZero = beta*p.W/alpha;
radiusSlope = betaSample/alpha;
coordinateGain = sqrt(diag(inverseP));
L0 = holdDerivativeBound(0, radiusZero, coordinateGain, p, ae, by, ce);
L1 = holdDerivativeBound(1, radiusZero+radiusSlope, coordinateGain, p, ae, by, ce);
L2 = holdDerivativeBound(2, radiusZero+2*radiusSlope, coordinateGain, p, ae, by, ce);
quadratic = (L2-2*L1+L0)/2;
linear = L1-L0-quadratic;
discriminant = (1-p.h*linear)^2 - 4*p.h^2*quadratic*L0;
holdBound = NaN;
radius = NaN;
inputBudget = NaN;
residual = NaN;
if discriminant >= 0 && (1-p.h*linear) > 0
    holdBound = 2*p.h*L0/(1-p.h*linear+sqrt(discriminant));
    holdBound = holdBound*(1+1e-7);
    radius = radiusZero + radiusSlope*holdBound;
    residual = holdBound-p.h*holdDerivativeBound(holdBound, radius, coordinateGain, p, ae, by, ce);
    inputBudget = p.U-radius*inputGain-holdBound;
end
nominalRateBound = p.k*p.sBound+(p.d+p.tau*p.k)*p.qBound ...
    +(1+p.tau*p.d)*p.jBound+p.tau*(p.omega^2*p.qBound+2*p.zeta*p.omega*p.jBound);
feasible = isfinite(inputBudget) && inputBudget > p.highStep*nominalRateBound && residual >= -1e-12;
cert = struct(feasible=feasible, A=A, B=B, P=P, alpha=alpha, ...
    betaDisturbance=beta, betaSample=betaSample, continuousRadius=radiusZero, ...
    radius=radius, holdBound=holdBound, holdSelfConsistencyResidual=residual, ...
    errorCoordinateBounds=radius*coordinateGain, inputErrorGain=inputGain, ...
    inputErrorBound=radius*inputGain, inputBudget=inputBudget, ...
    tightenedReferenceBudget=inputBudget-p.highStep*nominalRateBound, ...
    lyapunovResidual=norm(A'*P+P*A+eye(3), "fro"));
end

function bound = holdDerivativeBound(delta, radius, coordinateGain, p, ae, by, ce)
eBounds = radius*coordinateGain;
kr = p.k*(1+p.tau*(p.sBound+eBounds(2)));
ke = kr + abs(ae);
ks = p.d + p.tau*p.k;
ky = p.tau*p.k + abs(by);
bound = kr*p.sBound + ks*p.qBound + (1+p.tau*p.d)*p.jBound ...
    + p.tau*(p.omega^2*p.qBound+2*p.zeta*p.omega*p.jBound) ...
    + ke*eBounds(2) ...
    + ky*(p.kp*eBounds(1)+p.kv*eBounds(2)+eBounds(3)+p.W) ...
    + abs(ce)*(p.rho*eBounds(2)+p.ka*eBounds(3)+abs(p.kv-p.d)*p.W+delta/p.tau);
end

function planner = makePlanner(p, cert)
N = p.horizon;
Ac = [0, 1, 0; -p.omega^2, -2*p.zeta*p.omega, p.omega^2; 0, 0, 0];
Bc = [0; 0; 1];
transition = expm([Ac, Bc; zeros(1, 4)]*p.highStep);
Ad = transition(1:3, 1:3);
Bd = transition(1:3, 4);
beforeFree = zeros(3*N, 3);
beforeMap = zeros(3*N, N);
afterFree = zeros(3*N, 3);
afterMap = zeros(3*N, N);
currentFree = eye(3);
currentMap = zeros(3, N);
for index = 1:N
    indices = (index-1)*3+(1:3);
    beforeFree(indices, :) = currentFree;
    beforeMap(indices, :) = currentMap;
    currentFree = Ad*currentFree;
    currentMap = Ad*currentMap;
    currentMap(:, index) = currentMap(:, index) + Bd;
    afterFree(indices, :) = currentFree;
    afterMap(indices, :) = currentMap;
end
qRow = [-p.omega^2, -2*p.zeta*p.omega, p.omega^2];
jRow = -p.omega^2*[0, 1, 0] - 2*p.zeta*p.omega*qRow;
nominalRows = [1, 0, 0; 0, 1, 0; qRow; jRow];
inputWeights = [p.k, p.d+p.tau*p.k, 1+p.tau*p.d, p.tau];
signs = 2*(dec2bin(0:15, 4)-'0')-1;
caps = [p.rBound; p.sBound; p.qBound; p.jBound];
rateBounds = [p.sBound; p.qBound; p.jBound; ...
    p.omega^2*p.qBound+2*p.zeta*p.omega*p.jBound];
tightCaps = caps-rateBounds*p.highStep;
nominalRate = inputWeights*rateBounds;
inputCap = cert.inputBudget-nominalRate*p.highStep;
assert(all(tightCaps > 0) && inputCap > 0, "Reference intersample tightening is empty.");
constraintMatrix = zeros(48*N, N);
constraintFree = zeros(48*N, 3);
constraintLimits = zeros(48*N, 1);
constraintIndex = 0;
H = zeros(N);
gradientState = zeros(N, 3);
gradientTarget = zeros(N, N);
positionMap = zeros(N, N);
positionFree = zeros(N, 3);
for side = 1:2
    if side == 1
        free = beforeFree;
        mapped = beforeMap;
    else
        free = afterFree;
        mapped = afterMap;
    end
    for index = 1:N
        indices = (index-1)*3+(1:3);
        F = nominalRows*free(indices, :);
        G = nominalRows*mapped(indices, :);
        G(4, index) = G(4, index) + p.omega^2;
        inputF = (signs.*inputWeights)*F;
        inputG = (signs.*inputWeights)*G;
        count = 24;
        selected = constraintIndex+(1:count);
        constraintMatrix(selected, :) = [G; -G; inputG];
        constraintFree(selected, :) = [F; -F; inputF];
        constraintLimits(selected) = [tightCaps; tightCaps; inputCap*ones(16, 1)];
        constraintIndex = constraintIndex+count;
        if side == 2
            costTransform = [1, 0, 0, 0; 0, 1, 0, 0; p.k, p.d, 1, 0; inputWeights];
            costWeights = diag([1, 0.05, 0.01, 0.005]);
            if index == N
                costWeights(1, 1) = 3;
            end
            CF = costTransform*F;
            CG = costTransform*G;
            H = H + 2*CG'*costWeights*CG;
            gradientState = gradientState + 2*CG'*costWeights*CF;
            gradientTarget(:, index) = -2*CG'*costWeights(:, 1);
            positionMap(index, :) = G(1, :);
            positionFree(index, :) = F(1, :);
        end
    end
end
H = (H+H')/2;
[terminalRows, terminalLimits, terminalInfo] = makeTerminalSet(p, tightCaps, inputCap);
lastIndices = (N-1)*3+(1:3);
constraintMatrix = [constraintMatrix; terminalRows*afterMap(lastIndices, :)];
constraintFree = [constraintFree; terminalRows*afterFree(lastIndices, :)];
constraintLimits = [constraintLimits; terminalLimits];
planner = struct(H=H, gradientState=gradientState, gradientTarget=gradientTarget, ...
    A=constraintMatrix, free=constraintFree, limits=constraintLimits, ...
    positionMap=positionMap, positionFree=positionFree, ...
    terminalInfo=terminalInfo, options=optimoptions("quadprog", Algorithm="active-set", Display="off", ...
    ConstraintTolerance=1e-9, OptimalityTolerance=1e-9, MaxIterations=150));
end

function [rows, limits, info] = makeTerminalSet(p, caps, inputCap)
A2 = [0, 1; -p.omega^2, -2*p.zeta*p.omega];
P2 = lyap(A2', eye(2));
T = chol(P2);
gamma = norm(T*expm(A2*p.highStep)/T, 2);
faceCount = max(20, 2*ceil((pi/acos(gamma)+2)/2));
polygonFactor = cos(pi/faceCount);
assert(gamma < polygonFactor, "Terminal polygon is not invariant under the proposed backup.");
physicalRows = [1, 0; 0, 1; -p.omega^2, -2*p.zeta*p.omega; ...
    2*p.zeta*p.omega^3, 4*p.zeta^2*p.omega^2-p.omega^2];
gains = sqrt(sum((physicalRows/P2).*physicalRows, 2));
weights = [p.k, p.d+p.tau*p.k, 1+p.tau*p.d, p.tau];
centerLimit = min([0.28, 0.85*caps(1), 0.8*inputCap/p.k]);
radiusBounds = [(caps(1)-centerLimit)/gains(1); caps(2:4)./gains(2:4); ...
    (inputCap-p.k*centerLimit)/(weights*gains)];
radius = 0.95*polygonFactor*min(radiusBounds);
assert(radius > 0, "Terminal invariant set is empty.");
angles = (0:faceCount-1)'*(2*pi/faceCount);
directions = [cos(angles), sin(angles)];
rows = [directions*T*[1, 0, -1; 0, 1, 0]; 0, 0, 1; 0, 0, -1];
limits = [radius*ones(faceCount, 1); centerLimit; centerLimit];
info = struct(backup="nu=0, c constant", P=P2, contraction=gamma, ...
    polygonCosine=polygonFactor, faceCount=faceCount, radius=radius, ...
    centerLimit=centerLimit, invariantNormMargin=polygonFactor-gamma);
end

function updated = copyPlannerConstraints(updated, original)
updated.A = original.A;
updated.free = original.free;
updated.limits = original.limits;
end

function result = simulateCase(p, cert, planner, taskId, subdivisions, keepTrajectory)
steps = round(p.tStop/p.h);
highRatio = round(p.highStep/p.h);
time = (0:steps)'*p.h;
state = zeros(steps+1, 6);
state(1, :) = p.x0';
command = zeros(steps+1, 1);
control = zeros(steps+1, 1);
disturbance = zeros(steps+1, 1);
lyapunov = zeros(steps+1, 1);
targets = taskReference(time, taskId);
qpSuccess = true;
planningFeasible = true;
qpCalls = 0;
qpFailures = 0;
backupCalls = 0;
maximumQpResidual = 0;
previousSequence = zeros(p.horizon, 1);
holdError = 0;
formulaError = 0;
referenceViolation = 0;
nu = 0;
for index = 1:steps
    x = state(index, :)';
    if mod(index-1, highRatio) == 0
        z = x([5, 6, 4]);
        futureTime = time(index)+(1:p.horizon)'*p.highStep;
        futureTarget = taskReference(futureTime, taskId);
        gradient = planner.gradientState*z + planner.gradientTarget*futureTarget;
        limits = planner.limits-planner.free*z;
        warmSequence = [previousSequence(2:end); 0];
        [sequence, ~, exitFlag] = quadprog(planner.H, gradient, planner.A, ...
            limits, [], [], [], [], warmSequence, planner.options);
        qpCalls = qpCalls+1;
        if exitFlag <= 0
            qpSuccess = false;
            qpFailures = qpFailures+1;
            if max(planner.A*warmSequence-limits) <= 1e-8
                sequence = warmSequence;
                backupCalls = backupCalls+1;
            else
                planningFeasible = false;
                break
            end
        end
        maximumQpResidual = max(maximumQpResidual, max(planner.A*sequence-limits));
        previousSequence = sequence;
        nu = sequence(1);
    end
    disturbanceTime = floor((time(index)+1e-12)/p.disturbanceStep)*p.disturbanceStep;
    w = disturbanceSignal(disturbanceTime, taskId, p.W);
    [u, E, referenceData, canonicalU] = feedbackLaw(x, nu, p);
    formulaError = max(formulaError, abs(u-canonicalU));
    command(index) = nu;
    control(index) = u;
    disturbance(index) = w;
    lyapunov(index) = E'*cert.P*E;
    referenceViolation = max(referenceViolation, max([ ...
        abs(referenceData(1:4))-[p.rBound; p.sBound; p.qBound; p.jBound]; ...
        abs(referenceData(5))-cert.inputBudget]));
    heldU = max(-p.U, min(p.U, u));
    next = x;
    for substep = 1:subdivisions
        dt = p.h/subdivisions;
        k1 = combinedDynamics(next, heldU, nu, w, p);
        k2 = combinedDynamics(next+dt*k1/2, heldU, nu, w, p);
        k3 = combinedDynamics(next+dt*k2/2, heldU, nu, w, p);
        k4 = combinedDynamics(next+dt*k3, heldU, nu, w, p);
        next = next + dt*(k1+2*k2+2*k3+k4)/6;
        idealEnd = feedbackLaw(next, nu, p);
        holdError = max(holdError, abs(heldU-idealEnd));
    end
    state(index+1, :) = next';
end
if planningFeasible
    command(end) = command(end-1);
    disturbance(end) = disturbance(end-1);
    [control(end), finalError] = feedbackLaw(state(end, :)', command(end), p);
    lyapunov(end) = finalError'*cert.P*finalError;
    stateCost = (state(:, 1)-targets).^2 + 0.05*state(:, 2).^2 + 0.01*state(:, 3).^2;
    inputCost = 0.005*p.h*sum(min(p.U, abs(control(1:end-1))).^2);
    cost = trapz(time, stateCost)+inputCost;
else
    cost = Inf;
end
result = struct(taskId=taskId, qpSuccess=qpSuccess, planningFeasible=planningFeasible, ...
    qpCalls=qpCalls, qpFailures=qpFailures, backupCalls=backupCalls, ...
    maximumQpConstraintResidual=maximumQpResidual, cost=cost, ...
    rmsTaskError=sqrt(mean((state(:, 1)-targets).^2)), ...
    maxRawInput=max(abs(control)), maxInputViolation=max(0, max(abs(control))-p.U), ...
    maxLyapunovRatio=max(lyapunov)/cert.radius^2, ...
    maxHoldError=holdError, holdBound=cert.holdBound, ...
    maxReferenceViolation=max(0, referenceViolation), controlFormulaError=formulaError, ...
    time=[], state=[], command=[], control=[], disturbance=[], target=[], lyapunov=[]);
if keepTrajectory
    result.time = time;
    result.state = state;
    result.command = command;
    result.control = control;
    result.disturbance = disturbance;
    result.target = targets;
    result.lyapunov = lyapunov;
end
end

function dx = combinedDynamics(x, u, nu, w, p)
q = -p.omega^2*(x(5)-x(4))-2*p.zeta*p.omega*x(6);
dx = [x(2); x(3)-p.k*sin(x(1))-p.d*x(2)+w; ...
    (u-x(3))/p.tau; nu; x(6); q];
end

function [u, E, reference, canonicalU] = feedbackLaw(x, nu, p)
r = x(5);
s = x(6);
q = -p.omega^2*(r-x(4))-2*p.zeta*p.omega*s;
j = -p.omega^2*(s-nu)-2*p.zeta*p.omega*q;
e = x(1)-r;
y = x(2)-s;
b = x(3)-p.k*sin(x(1))-p.d*x(2);
desiredActuator = p.k*sin(x(1))+p.d*x(2)+q-p.kp*e-p.kv*y;
actuatorError = x(3)-desiredActuator;
desiredDerivative = p.k*cos(x(1))*x(2)+(p.d-p.kv)*b+j-p.kp*y+p.kv*q;
u = x(3)+p.tau*(desiredDerivative-p.ka*actuatorError-p.rho*y);
E = [e; y; actuatorError];
nominalU = p.k*sin(r)+p.d*s+q+p.tau*(p.k*cos(r)*s+p.d*q+j);
reference = [r; s; q; j; nominalU];
ae = -p.kp*(1+p.tau*(p.d-p.kv));
by = p.d-p.kv-p.tau*((p.d-p.kv)*p.kv+p.kp+p.rho);
ce = 1+p.tau*(p.d-p.kv-p.ka);
canonicalU = p.k*sin(r+e)+p.d*s+q ...
    + p.tau*(p.k*cos(r+e)*(s+y)+p.d*q+j)+ae*e+by*y+ce*actuatorError;
end

function target = taskReference(time, taskId)
switch taskId
    case 1
        target = 0.24*sin(2*pi*time/9);
    case 2
        target = 0.23*(time >= 1)-0.43*(time >= 4.2)+0.20*(time >= 7);
    case 3
        target = 0.17*sin(2*pi*time/7)+0.06*sin(2*pi*time/3.7);
    otherwise
        error("Unknown declared task.");
end
end

function w = disturbanceSignal(time, taskId, limit)
switch taskId
    case 1
        w = limit*sin(2.1*time);
    case 2
        w = limit*(2*(sin(1.4*time) >= 0)-1);
    case 3
        w = limit*(0.6*sin(3.3*time+0.7)+0.4*cos(0.8*time));
    otherwise
        error("Unknown declared disturbance scenario.");
end
end

function stripped = stripTrajectory(result)
stripped = rmfield(result, ["time", "state", "command", "control", "disturbance", "target", "lyapunov"]);
end

function result = verifySimulink(outputDir, params, trajectory)
modelName = "codesign_nonlinear_closed_loop";
modelDir = fullfile(outputDir, "models");
modelPath = fullfile(modelDir, modelName+".slx");
nuData = [trajectory.time, trajectory.command];
wData = [trajectory.time, trajectory.disturbance];
if bdIsLoaded(modelName) || isfile(modelPath)
    if ~bdIsLoaded(modelName)
        load_system(modelPath);
    end
    assert(strcmpi(string(get_param(modelName, "FileName")), modelPath), ...
        "A different model is using the reserved prototype name.");
    simIn = Simulink.SimulationInput(modelName);
    fields = {"k", "d", "tau", "U", "h", "x0", "tStop", "omega", "zeta", "kp", "kv", "ka", "rho"};
    workspace = get_param(modelName, "ModelWorkspace");
    for index = 1:numel(fields)
        field = fields{index};
        simIn = simIn.setVariable("cl_"+field, params.(field), Workspace=modelName);
        assignin(workspace, "cl_"+field, params.(field));
    end
    simIn = simIn.setVariable("nuData", nuData, Workspace=modelName);
    simIn = simIn.setVariable("wData", wData, Workspace=modelName);
    assignin(workspace, "nuData", nuData);
    assignin(workspace, "wData", wData);
    simIn = simIn.setModelParameter(FixedStep=num2str(params.h, 17), StopTime=num2str(params.tStop, 17));
    set_param(modelName, "FixedStep", num2str(params.h, 17), "StopTime", num2str(params.tStop, 17));
    save_system(modelName, modelPath);
else
    [modelPath, simIn] = buildNonlinearClosedLoopModel(modelDir, params, nuData, wData, false);
end
simOut = sim(simIn);
simTime = simOut.pLog.Time;
simState = [simOut.pLog.Data, simOut.vLog.Data, simOut.aLog.Data, ...
    simOut.cLog.Data, simOut.z1Log.Data, simOut.z2Log.Data];
aligned = interp1(trajectory.time, trajectory.state, simTime, "linear");
stateError = max(abs(simState-aligned), [], "all");
assert(stateError < 1e-7, "Native Simulink and independent RK4 trajectories disagree.");
timeError = max(abs(simTime-trajectory.time));
nuError = max(abs(simOut.nuLog.Data-trajectory.command));
disturbanceError = max(abs(simOut.wLog.Data-trajectory.disturbance));
rawControlError = max(abs(simOut.uRawLog.Data-trajectory.control));
assert(nuError < 1e-10 && disturbanceError < 1e-10 && rawControlError < 1e-7, ...
    "Native Simulink input samples disagree with the independent implementation.");
open_system(modelName);
result = struct(modelPath=modelPath, samples=numel(simTime), maximumStateDifference=stateError, ...
    maximumTimeDifference=timeError, maximumCommandDifference=nuError, ...
    maximumDisturbanceDifference=disturbanceError, maximumControlDifference=rawControlError, ...
    maximumRawInput=max(abs(simOut.uRawLog.Data)), ...
    scope="Native low-level feedback closed loop; high-level command sequence from matching MATLAB MPC rollout");
save(fullfile(outputDir, "simulink-output.mat"), "simOut");
end

function renderResults(outputDir, report, trajectory)
fig = figure(Name="Nonlinear response co-design audit", Color="white", Position=[100, 100, 1100, 800]);
layout = tiledlayout(fig, 3, 1, TileSpacing="compact", Padding="compact");
nexttile(layout);
plot(trajectory.time, trajectory.target, "k--", LineWidth=1.4);
hold on
plot(trajectory.time, trajectory.state(:, 5), Color=[0.1, 0.45, 0.75], LineWidth=1.5);
plot(trajectory.time, trajectory.state(:, 1), Color=[0.85, 0.3, 0.12], LineWidth=1.3);
ylabel("Position");
legend("Task", "MPC reference", "Physical output", Location="best");
grid on
nexttile(layout);
plot(trajectory.time, trajectory.control, LineWidth=1.3);
hold on
yline(report.bestParameters.U, "r--");
yline(-report.bestParameters.U, "r--");
ylabel("Physical input");
grid on
nexttile(layout);
plot(trajectory.time, trajectory.lyapunov/report.certificate.radius^2, LineWidth=1.4);
hold on
yline(1, "r--", "Certified envelope");
ylabel("V / R^2");
xlabel("Time (s)");
grid on
title(layout, "Nonlinear realization under actuator and input limits", FontSize=14, Color="black");
subtitle(layout, "Held-out task; matched-class joint-design gain = 0", FontSize=11, Color="black");
axesHandles = findall(fig, Type="axes");
for axesIndex = 1:numel(axesHandles)
    axesHandles(axesIndex).Toolbar.Visible = "off";
    axesHandles(axesIndex).Color = "white";
    axesHandles(axesIndex).XColor = [0.15, 0.15, 0.15];
    axesHandles(axesIndex).YColor = [0.15, 0.15, 0.15];
end
set(findall(fig, Type="text"), Color="black");
set(findall(fig, Type="legend"), Color="white", TextColor="black", EdgeColor=[0.6, 0.6, 0.6]);
exportgraphics(fig, fullfile(outputDir, "nonlinear-prototype.png"), Resolution=180);
savefig(fig, fullfile(outputDir, "nonlinear-prototype.fig"));
end

function writeJson(filePath, value)
fileId = fopen(filePath, "w", "n", "UTF-8");
assert(fileId >= 0, "Cannot create the result JSON.");
cleanup = onCleanup(@() fclose(fileId));
fprintf(fileId, "%s\n", jsonencode(value, PrettyPrint=true));
end
