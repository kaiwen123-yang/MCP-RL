function nextState = learningPilotPlantStep(state, action, disturbance, cfg)
%LEARNINGPILOTPLANTSTEP Nonlinear actuator plant, ZOH and RK4/two substeps.
%   State is [p;v;a]. Disturbance remains the supplied exogenous signal; it is
%   not clipped to the nominal W domain. Actual actuator command is limited.
nextState = single(state(:));
action = min(max(single(action), -single(cfg.U)), single(cfg.U));
disturbance = single(disturbance);
step = single(cfg.dt/2);
for index = 1:2
    first = derivative(nextState, action, disturbance, cfg);
    second = derivative(nextState + single(0.5)*step*first, action, disturbance, cfg);
    third = derivative(nextState + single(0.5)*step*second, action, disturbance, cfg);
    fourth = derivative(nextState + step*third, action, disturbance, cfg);
    nextState = nextState + (step/single(6))*(first + single(2)*second ...
        + single(2)*third + fourth);
end
end

function value = derivative(state, action, disturbance, cfg)
p = state(1);
v = state(2);
a = state(3);
acceleration = a-single(cfg.k)*sin(p)-single(cfg.d)*v ...
    -single(cfg.drag)*v*abs(v)+disturbance;
actuatorDerivative = (action-a)/single(cfg.tau);
value = [v; acceleration; actuatorDerivative];
end
