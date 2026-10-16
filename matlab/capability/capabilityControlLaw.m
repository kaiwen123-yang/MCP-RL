function [raw,f,g] = capabilityControlLaw(x,c,theta,cfg)
%CAPABILITYCONTROLLAW Columns are independent trajectories/candidates.
f = cfg.dv*x(3,:)+cfg.kv*(c-x(3,:));
shape = x(2,:).^2./(cfg.q0^2+x(2,:).^2);
g = -theta(1)*sin(x(2,:))-(theta(2)+theta(3)*shape).*x(4,:);
raw = [f+g;f-g];
end
