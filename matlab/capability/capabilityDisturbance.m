function w = capabilityDisturbance(t,cfg)
%CAPABILITYDISTURBANCE Hidden disturbance, held over each low-level sample.
switch cfg.caseId
    case 1 % Design: isolated positive kick and later negative kick.
        w = 1.4*(t>=1.2 && t<1.5)-0.9*(t>=3.5 && t<3.8);
    case 2 % Design: initial displacement, followed by a sustained load.
        w = 0.55*(t>=2 && t<2.8);
    case 3 % Held-out: opposite sign, changed timing and duration.
        w = -1.5*(t>=0.9 && t<1.25)+1.1*(t>=4 && t<4.25);
    case 4 % Held-out: actuator asymmetry plus two same-sign kicks.
        w = 1.2*(t>=1.7 && t<2.1)+1.0*(t>=3.8 && t<4.05);
    case 5 % Held-out: continuous disturbance with command switches.
        w = 0.25*sin(2.3*t)+0.8*(t>=2.6 && t<2.9);
    otherwise % Nominal response probe.
        w = 0;
end
end
