function action = learningPilotActor(state, command, net)
%LEARNINGPILOTACTOR Actual frozen 12-32-32-1 full-action policy.
%   Matches learning.py execution_features, including its intentionally fixed
%   features. No response design, virtual state, global target or teacher is
%   added to this deployed control law. Net tensors use PyTorch Linear layout.
state = single(state(:));
command = single(command);
p = state(1);
v = state(2);
a = state(3);
features = [p/1.5; v/2; a/3; command; (p-command)/1.5; ...
    sin(p); cos(p)-1; v*abs(v)/4; (a-sin(p)-0.2*v)/3; ...
    p*v/3; command*v/2; single(1)];
hidden1 = tanh(net.W1*features + net.b1);
hidden2 = tanh(net.W2*hidden1 + net.b2);
rawAction = net.W3*hidden2 + net.b3;
action = single(1.2)*tanh(rawAction);
end
