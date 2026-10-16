function capabilityControllerBlock(block)
%CAPABILITYCONTROLLERBLOCK Sampled feedback with an online macro-step planner.
%   Dialog parameters are cfg and theta. Input is actual [p;q;v;r;a1;a2].
%   Outputs are raw actuator command, macro command, held disturbance, and
%   [controller evaluations; macro replans]. Repeated Outputs calls at the
%   same sample reuse the cache, so they do not repeat an optimization.

block.NumDialogPrms = 2;
block.DialogPrmsTunable = {'Nontunable', 'Nontunable'};
block.NumInputPorts = 1;
block.NumOutputPorts = 4;
block.SetPreCompInpPortInfoToDynamic;
block.SetPreCompOutPortInfoToDynamic;
block.InputPort(1).Dimensions = 6;
block.InputPort(1).DatatypeID = 0;
block.InputPort(1).Complexity = 'Real';
block.InputPort(1).DirectFeedthrough = true;
dimensions = [2, 1, 1, 2];
for index = 1:4
    block.OutputPort(index).Dimensions = dimensions(index);
    block.OutputPort(index).DatatypeID = 0;
    block.OutputPort(index).Complexity = 'Real';
end
cfg = block.DialogPrm(1).Data;
block.SampleTimes = [double(cfg.dt), 0];
block.SimStateCompliance = 'DefaultSimState';
block.RegBlockMethod('PostPropagationSetup', @setupWork);
block.RegBlockMethod('InitializeConditions', @initialize);
block.RegBlockMethod('Outputs', @outputs);
block.RegBlockMethod('Update', @update);
end

function setupWork(block)
cfg = block.DialogPrm(1).Data;
block.NumDworks = 3;
names = {'Plan', 'NextPlan', 'OutputCache'};
dimensions = [double(cfg.H), double(cfg.H), 7];
for index = 1:3
    block.Dwork(index).Name = names{index};
    block.Dwork(index).Dimensions = dimensions(index);
    block.Dwork(index).DatatypeID = 0;
    block.Dwork(index).Complexity = 'Real';
    block.Dwork(index).UsedAsDiscState = (index == 1);
end
end

function initialize(block)
cfg = block.DialogPrm(1).Data;
block.Dwork(1).Data = zeros(cfg.H, 1);
block.Dwork(2).Data = zeros(cfg.H, 1);
% Cache: raw(1:2), command, disturbance, sample index, call/replan counts.
block.Dwork(3).Data = [0; 0; 0; 0; -1; 0; 0];
end

function outputs(block)
cfg = block.DialogPrm(1).Data;
theta = block.DialogPrm(2).Data;
cache = block.Dwork(3).Data;
sampleIndex = round(block.CurrentTime / cfg.dt);
if sampleIndex ~= cache(5)
    sampleTime = sampleIndex * cfg.dt;
    [raw, command, nextPlan, info] = capabilityController( ...
        block.InputPort(1).Data, sampleTime, theta, cfg, block.Dwork(1).Data);
    disturbance = capabilityDisturbance(sampleTime, cfg);
    block.Dwork(2).Data = nextPlan(:);
    cache(1:4) = [raw(:); command; disturbance];
    cache(5) = sampleIndex;
    cache(6) = cache(6) + 1;
    cache(7) = cache(7) + double(info.replanned);
    block.Dwork(3).Data = cache;
end
block.OutputPort(1).Data = cache(1:2);
block.OutputPort(2).Data = cache(3);
block.OutputPort(3).Data = cache(4);
block.OutputPort(4).Data = cache(6:7);
end

function update(block)
block.Dwork(1).Data = block.Dwork(2).Data;
end
