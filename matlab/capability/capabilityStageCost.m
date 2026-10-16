function cost = capabilityStageCost(x,u,t,cfg)
%CAPABILITYSTAGECOST Identical physical task objective for all comparisons.
[pr,vr] = capabilityReference(t);
w = cfg.weights;
cost = w(1)*(x(1,:)-pr).^2+w(2)*x(2,:).^2+ ...
    w(3)*(x(3,:)-vr).^2+w(4)*x(4,:).^2+w(5)*sum(u.^2,1);
end
