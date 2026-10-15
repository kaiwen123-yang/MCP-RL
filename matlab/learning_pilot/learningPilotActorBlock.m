function learningPilotActorBlock(block)
%LEARNINGPILOTACTORBLOCK Level-2 discrete full-action neural controller.
block.NumDialogPrms = 2;
block.DialogPrmsTunable = {'Nontunable', 'Nontunable'};
block.NumInputPorts = 2;
block.NumOutputPorts = 1;
block.SetPreCompInpPortInfoToDynamic;
block.SetPreCompOutPortInfoToDynamic;
block.InputPort(1).Dimensions = 3;
block.InputPort(2).Dimensions = 1;
for index = 1:2
    block.InputPort(index).DatatypeID = 1;
    block.InputPort(index).Complexity = 'Real';
    block.InputPort(index).DirectFeedthrough = true;
end
block.OutputPort(1).Dimensions = 1;
block.OutputPort(1).DatatypeID = 1;
block.OutputPort(1).Complexity = 'Real';
block.SampleTimes = [double(block.DialogPrm(2).Data), 0];
block.SimStateCompliance = 'DefaultSimState';
block.RegBlockMethod('Outputs', @outputs);
end

function outputs(block)
block.OutputPort(1).Data = learningPilotActor(block.InputPort(1).Data, ...
    block.InputPort(2).Data, block.DialogPrm(1).Data);
end
