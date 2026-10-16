function x = capabilityIntegrate(x,u,w,cfg,nSubsteps)
%CAPABILITYINTEGRATE RK4 with sampled input and disturbance held constant.
h = cfg.dt/nSubsteps;
for j = 1:nSubsteps
    k1 = capabilityDerivative(x,u,w,cfg);
    k2 = capabilityDerivative(x+h*k1/2,u,w,cfg);
    k3 = capabilityDerivative(x+h*k2/2,u,w,cfg);
    k4 = capabilityDerivative(x+h*k3,u,w,cfg);
    x = x+h*(k1+2*k2+2*k3+k4)/6;
end
end
