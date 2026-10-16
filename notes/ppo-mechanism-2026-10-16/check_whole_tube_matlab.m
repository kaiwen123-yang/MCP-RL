% Independent DARE and quadratic-vertex check. No training or simulation.
% This is a floating-point cross-check, not interval arithmetic certification.
wholeTubeMatlabReport = recomputeWholeTube();
disp(wholeTubeMatlabReport.comparison);

function report = recomputeWholeTube()
    folder = fileparts(mfilename('fullpath'));
    pythonReport = jsondecode(fileread(fullfile(folder, 'whole-tube-check.json')));
    h = 0.02;
    tau = 0.12;
    rho = 0.98;
    delta = 4e-6;
    disturbanceBound = 0.04;
    A = [1, h, 0; -h, 1-0.2*h, h; 0, 0, 1-h/tau];
    B = [0; 0; h/tau];
    E = [0; h; 0];
    Q = diag([2, 0.5, 0.02]);
    R = 0.05;
    [storage, ~, K] = dare(A, B, Q, R);
    P = storage / trace(storage);
    tubeBudget = delta / (1-rho);
    vertices = zeros(4, 4);
    index = 0;
    for alpha = [0, 0.204]
        for beta = [0, 0.2]
            index = index + 1;
            M = A - B*K;
            M(2, 1) = M(2, 1) + h*alpha;
            M(2, 2) = M(2, 2) - h*beta;
            Qi = rho*P - M'*P*M;
            b = M'*P*E*disturbanceBound;
            c = disturbanceBound^2*(E'*P*E);
            maxExcess = c + b'*(Qi\b);
            vertices(index, :) = [alpha, beta, min(eig(Qi)), maxExcess];
        end
    end
    inputCorrection = sqrt(tubeBudget*K*(P\K'));
    stateMargins = sqrt(tubeBudget*diag(P\eye(3)));
    assert(min(vertices(:, 3)) > 0);
    assert(max(vertices(:, 4)) < delta);
    assert(1+inputCorrection < 1.2);
    pythonExcess = [pythonReport.vertices.max_excess_before_delta]';
    comparison = struct( ...
        'max_P_difference', max(abs(P-pythonReport.P), [], 'all'), ...
        'max_K_difference', max(abs(K-pythonReport.K(:)')), ...
        'max_vertex_excess_difference', max(abs(vertices(:, 4)-pythonExcess)), ...
        'input_correction_difference', abs(inputCorrection-pythonReport.max_input_correction), ...
        'state_margins_difference', max(abs(stateMargins-pythonReport.state_margins(:))));
    assert(comparison.max_P_difference < 1e-10);
    assert(comparison.max_K_difference < 1e-9);
    assert(comparison.max_vertex_excess_difference < 1e-12);
    report = struct('kind', 'Independent MATLAB floating-point algebra check; no training', ...
        'matlab_version', version, 'P', P, 'K', K, ...
        'vertices_columns', {{'alpha', 'beta', 'min_Q_eigenvalue', 'max_excess_before_delta'}}, ...
        'vertices', vertices, 'max_input_correction', inputCorrection, ...
        'state_margins', stateMargins, 'comparison', comparison, ...
        'boundary', 'Not interval arithmetic certification or full MPC recursive feasibility');
    fileId = fopen(fullfile(folder, 'whole-tube-matlab-check.json'), 'w');
    fileCloser = onCleanup(@() fclose(fileId));
    fprintf(fileId, '%s\n', jsonencode(report, PrettyPrint=true));
end
