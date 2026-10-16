function [pRef,vRef] = capabilityReference(t)
%CAPABILITYREFERENCE Common preview, independent of feedback and disturbance.
knots = [0 1.8 3 4.6];
speeds = [0.65 0.3 0.8 0.2];
pRef = zeros(size(t));
vRef = speeds(1)*ones(size(t));
for j = 1:numel(knots)
    if j < numel(knots)
        duration = max(0,min(t-knots(j),knots(j+1)-knots(j)));
    else
        duration = max(0,t-knots(j));
    end
    pRef = pRef+speeds(j)*duration;
    vRef(t >= knots(j)) = speeds(j);
end
end
