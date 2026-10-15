function learningPilotPlantBlock(block)
%LEARNINGPILOTPLANTBLOCK Level-2 RK4 plant map; state memory is a Unit Delay.
block.NumDialogPrms = 1;
block.DialogPrmsTunable = {'Nontunable'};
block.NumInputPorts = 3;
block.NumOutputPorts = 1;
block.SetPreCompInpPortInfoToDynamic;
block.SetPreCompOutPortInfoToDynamic;
for index = 1:3
    block.InputPort(index).Dimensions = 1;
    block.InputPort(index).DatatypeID = 1;
    block.InputPort(index).Complexity = 'Real';
    block.InputPort(index).DirectFeedthrough = true;
end
block.InputPort(1).Dimensions = 3;
block.OutputPort(1).Dimensions = 3;
block.OutputPort(1).DatatypeID = 1;
block.OutputPort(1).Complexity = 'Real';
cfg = block.DialogPrm(1).Data;
block.SampleTimes = [double(cfg.dt), 0];
block.SimStateCompliance = 'DefaultSimState';
block.RegBlockMethod('Outputs', @outputs);
end

function outputs(block)
block.OutputPort(1).Data = learningPilotPlantStep(block.InputPort(1).Data, ...
    block.InputPort(2).Data, block.InputPort(3).Data, block.DialogPrm(1).Data);
end
