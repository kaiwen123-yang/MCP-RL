function result = learningPilotRealizationCheck(initialState, states, actions, commands, rawDesign, cfg)
%LEARNINGPILOTREALIZATIONCHECK Recompute the pilot's two empirical constraints.
%   The predictor restarts from actual state every ten steps; the actual
%   plant does not restart. Input mismatch uses the pre-step nominal input;
%   response mismatch uses the post-step physical and response states.
state = single(initialState(:));
design = single([1.2; 0.7; 3; 0]) + single([1.8; 0.7; 11; 0.5]) ...
    ./ (single(1)+exp(-single(rawDesign(:))));
steps = numel(actions);
tracking = zeros(steps, 1, 'single');
inputMismatch = zeros(steps, 1, 'single');
reference = zeros(3, 1, 'single');
for index = 1:steps
    command = single(commands(index));
    if mod(index-1, 10) == 0
        nominalAcceleration = state(3)-single(cfg.k)*sin(state(1)) ...
            -single(cfg.d)*state(2)-single(cfg.drag)*state(2)*abs(state(2));
        reference = [state(1); state(2); nominalAcceleration];
    end
    derivative = referenceDerivative(reference, command, design);
    r = reference(1);
    s = reference(2);
    q = reference(3);
    actuator = single(cfg.k)*sin(r)+single(cfg.d)*s ...
        +single(cfg.drag)*s*abs(s)+q;
    actuatorDerivative = single(cfg.k)*cos(r)*s ...
        +(single(cfg.d)+single(2)*single(cfg.drag)*abs(s))*q+derivative(3);
    nominalInput = actuator+single(cfg.tau)*actuatorDerivative;
    h = single(cfg.dt/2);
    for substep = 1:2
        first = referenceDerivative(reference, command, design);
        second = referenceDerivative(reference+single(0.5)*h*first, command, design);
        third = referenceDerivative(reference+single(0.5)*h*second, command, design);
        fourth = referenceDerivative(reference+h*third, command, design);
        reference = reference+(h/single(6))*(first+single(2)*second+single(2)*third+fourth);
    end
    state = single(states(index, :))';
    tracking(index) = (state(1)-reference(1))^2 ...
        +single(0.05)*(state(2)-reference(2))^2;
    inputMismatch(index) = (single(actions(index))-nominalInput)^2;
end
result = struct('tracking_q1_mean', double(mean(tracking)), ...
    'input_q2_mean', double(mean(inputMismatch)), ...
    'scope', 'One frozen evaluation episode; empirical means, not a domain-wide certificate', ...
    'macro_steps', 10, 'response_jerk_limit', 8, ...
    'response_ranges', [1.2 3; 0.7 1.4; 3 14; 0 0.5]);
end

function derivative = referenceDerivative(state, command, design)
r = state(1);
s = state(2);
q = state(3);
omega = design(1)/(single(1)+design(4)*s^2);
jerk = design(3)*(omega^2*(command-r)-single(2)*design(2)*omega*s-q);
derivative = [s; q; min(max(jerk, single(-8)), single(8))];
end
