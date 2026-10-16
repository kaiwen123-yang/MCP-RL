function artifacts = plotCapabilityStudy(root)
%PLOTCAPABILITYSTUDY Native MATLAB figures from the completed frozen study.
%   ARTIFACTS = plotCapabilityStudy(ROOT) reads study.mat without running a
%   controller, optimizer, simulation or training. It leaves the two figures
%   open, exports public PNGs, and saves editable FIGs in the ignored cache.
%   Each evaluation point is one deterministic case, not a random replicate.

if nargin == 0
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
cacheDir = fullfile(root, 'work', 'capability-probe-2026-10-16');
data = load(fullfile(cacheDir, 'study.mat'), 'result', 'evaluation', 'probes');
assert(isequal(size(data.probes), [2, 3]) && ...
    isequal(size(data.evaluation), [3, 4]), 'Capability:FigureData', ...
    'Expected the complete 2-by-3 probes and 3-by-4 held-out evaluations.');
publicDir = fullfile(root, 'notes', 'capability-probe-2026-10-16', 'figures');
if ~isfolder(publicDir), mkdir(publicDir); end

% Okabe-Ito colors and redundant line/marker encodings.
palette = [0.12 0.12 0.12; 0.000 0.447 0.698; ...
    0.835 0.369 0.000; 0.580 0.360 0.680];
styles = {'-', '--', ':', '-.'};
markers = {'o', 's', '^', 'd'};
probes = data.probes;
probeLabels = cell(1, 3);
for j = 1:3
    probeLabels{j} = sprintf('Nonlinear damping = %g', probes{1,j}.theta(3));
end

% Profile only quantities used by the plots. No statistical uncertainty is
% inferred from these single deterministic trajectories.
profile.probeRawPeakOverLimit = zeros(2, 3);
profile.probeClippedPercent = zeros(2, 3);
profile.nominalMaximumStateDifference = 0;
qAll = []; vAll = [];
for mode = 1:2
    for j = 1:3
        run = probes{mode,j};
        assert(all(isfinite(run.x), 'all') && all(isfinite(run.raw), 'all'), ...
            'Capability:NonfiniteFigureData', 'Probe data contain nonfinite values.');
        profile.probeRawPeakOverLimit(mode,j) = max(abs(run.raw), [], 'all') / run.cfg.U;
        profile.probeClippedPercent(mode,j) = 100 * mean(any(abs(run.raw) > run.cfg.U + 1e-9, 2));
        qAll = [qAll; run.x(:,2)]; %#ok<AGROW>
        vAll = [vAll; run.x(:,3)]; %#ok<AGROW>
        if mode == 1
            profile.nominalMaximumStateDifference = max( ...
                profile.nominalMaximumStateDifference, ...
                max(abs(run.x - probes{1,1}.x), [], 'all'));
        end
    end
end
qBounds = paddedBounds(qAll, 0.10);
vBounds = [0, 1.08 * max(vAll)];
demandCeiling = 1.12 * max(profile.probeRawPeakOverLimit, [], 'all');

fig1 = newFigure('Capability: matched fixed-command probes', [10.8, 7.8]);
layout1 = tiledlayout(fig1, 2, 3, 'TileSpacing', 'loose', 'Padding', 'loose');
layout1.Units = 'normalized';
layout1.OuterPosition = [0, 0.085, 1, 0.83];
probeTheta = probes{1,1}.theta;
addHeader(fig1, 'Matched fixed-command probes | classical feedback, no RL');
addCaption(fig1, sprintf(['Same k_p = %g and k_d = %g; fixed command c = %.2f. ' ...
    'Only nonlinear damping changes. Rows share axis limits.'], ...
    probeTheta(1), probeTheta(2), probes{1,1}.command(1)));
probeHandles = gobjects(1, 3);
for mode = 1:2
    axQ = nexttile(layout1, (mode-1)*3+1);
    axU = nexttile(layout1, (mode-1)*3+2);
    axV = nexttile(layout1, (mode-1)*3+3);
    for ax = [axQ, axU, axV]
        styleAxes(ax);
        xlabel(ax, 'Time (s)');
        xlim(ax, [0, probes{mode,1}.cfg.T]);
    end
    for j = 1:3
        run = probes{mode,j};
        h = plot(axQ, run.t, run.x(:,2), 'Color', palette(j,:), ...
            'LineStyle', styles{j}, 'LineWidth', 1.6, 'Marker', markers{j}, ...
            'MarkerSize', 3.5, 'MarkerIndices', (8+8*j):55:numel(run.t));
        if mode == 1, probeHandles(j) = h; end
        plotHeld(axU, run, max(abs(run.raw), [], 2)/run.cfg.U, ...
            palette(j,:), styles{j});
        plot(axV, run.t, run.x(:,3), 'Color', palette(j,:), ...
            'LineStyle', styles{j}, 'LineWidth', 1.6, 'Marker', markers{j}, ...
            'MarkerSize', 3.5, 'MarkerIndices', (8+8*j):55:numel(run.t));
    end
    ylabel(axQ, 'Internal angle q (rad)');
    ylabel(axU, {'Raw actuator demand', 'max_i |u_i^{raw}| / U'});
    ylabel(axV, 'Forward speed v (normalized / s)');
    ylim(axQ, qBounds);
    ylim(axU, [0, demandCeiling]);
    ylim(axV, vBounds);
    yline(axU, 1, '-.', 'Clipping limit', 'Color', 'black', ...
        'LabelHorizontalAlignment', 'right', 'FontSize', 8, 'HandleVisibility', 'off');
    yline(axV, probes{mode,1}.command(1), '-.', 'Command', ...
        'Color', 'black', 'LabelHorizontalAlignment', 'right', ...
        'FontSize', 8, 'HandleVisibility', 'off');
    if mode == 1
        panelTitle(axQ, 'a  Nominal: internal response');
        panelTitle(axU, 'b  Nominal: input resource');
        panelTitle(axV, 'c  Nominal: forward response');
        text(axQ, 0.03, 0.95, sprintf(['q(0) = r(0) = 0; w = 0\n' ...
            'Max state difference = %.2g'], profile.nominalMaximumStateDifference), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 8.5);
    else
        panelTitle(axQ, 'd  Perturbed: internal response');
        panelTitle(axU, 'e  Perturbed: input resource');
        panelTitle(axV, 'f  Perturbed: forward response');
        text(axQ, 0.98, 0.95, sprintf('q(0) = %.2f rad\nr(0) = %.2f rad/s; w = 0', ...
            probes{mode,1}.x(1,2), probes{mode,1}.x(1,4)), ...
            'Units', 'normalized', 'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'top', 'FontSize', 8.5);
    end
    peak = profile.probeRawPeakOverLimit(mode,:);
    clipped = profile.probeClippedPercent(mode,:);
    text(axU, 0.98, 0.95, sprintf(['Damping order: 0 / 3 / 8\n' ...
        'peak/U: %.2f / %.2f / %.2f\nclipped time: %.1f / %.1f / %.1f%%'], ...
        peak(1), peak(2), peak(3), clipped(1), clipped(2), clipped(3)), ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', 'FontSize', 8.2);
end
probeLegend = legend(probeHandles, probeLabels, 'Orientation', 'horizontal', ...
    'NumColumns', 3, 'Box', 'off', 'FontSize', 9, ...
    'TextColor', 'black', 'Color', 'white');
probeLegend.Layout.Tile = 'south';

% Keep all twelve held-out costs visible; cases are distinct tests rather
% than IID samples. There are no means, error bars or significance claims.
evaluation = data.evaluation;
cost = zeros(3, 4);
caseIds = zeros(1, 3);
for k = 1:3
    caseIds(k) = evaluation{k,1}.cfg.caseId;
    for g = 1:4
        cost(k,g) = evaluation{k,g}.metrics.cost;
    end
end
assert(all(isfinite(cost), 'all'), 'Capability:NonfiniteFigureData', ...
    'Held-out costs contain nonfinite values.');
profile.evaluationCost = cost;
profile.evaluationCaseIds = caseIds;
linear = data.result.design.selectedLinear;
extended = data.result.design.selectedExtended;
groupLabels = {sprintf('Linear + MPC (%.2f s)', ...
    evaluation{1,1}.cfg.dt * evaluation{1,1}.cfg.planEvery), ...
    sprintf('Extended + MPC (%.2f s)', ...
    evaluation{1,2}.cfg.dt * evaluation{1,2}.cfg.planEvery), ...
    sprintf('Linear + fast MPC (%.2f s)', ...
    evaluation{1,3}.cfg.dt * evaluation{1,3}.cfg.planEvery), ...
    'Extended + linear-command replay'};
groupColors = palette([1,3,2,4], :);
fig2 = newFigure('Capability: held-out task diagnostics', [10.8, 8.0]);
layout2 = tiledlayout(fig2, 2, 2, 'TileSpacing', 'loose', 'Padding', 'loose');
layout2.Units = 'normalized';
layout2.OuterPosition = [0, 0.085, 1, 0.83];
addHeader(fig2, 'Held-out task checks | classical feedback, no RL');
addCaption(fig2, sprintf(['Selected on design cases 1-2: linear [%g, %g, %g]; ' ...
    'extended [%g, %g, %g]. Different k_p; this is not a matched-linearization comparison.'], ...
    linear(1), linear(2), linear(3), extended(1), extended(2), extended(3)));
axCost = nexttile(layout2, 1);
styleAxes(axCost);
groupHandles = gobjects(1, 4);
offset = [-0.24, -0.08, 0.08, 0.24];
for g = 1:4
    groupHandles(g) = plot(axCost, (1:3)+offset(g), cost(:,g), ...
        'LineStyle', 'none', 'Marker', markers{g}, 'MarkerSize', 7, ...
        'Color', groupColors(g,:), 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
end
xlim(axCost, [0.5, 3.5]);
xticks(axCost, 1:3);
xticklabels(axCost, compose('Case %d', caseIds));
ylim(axCost, [0, 1.16*max(cost, [], 'all')]);
ylabel(axCost, 'Integrated task cost J (normalized)');
xlabel(axCost, 'Held-out scenario (one deterministic run per group)');
panelTitle(axCost, 'a  Actual cost: all held-out cases');

axQ = nexttile(layout2, 2);
axC = nexttile(layout2, 3);
axU = nexttile(layout2, 4);
for ax = [axQ, axC, axU]
    styleAxes(ax);
    xlabel(ax, 'Time (s)');
    xlim(ax, [0, evaluation{1,1}.cfg.T]);
end
qAll = []; cAll = []; demandMaximum = 1;
for g = 1:4
    run = evaluation{1,g};
    plot(axQ, run.t, run.x(:,2), 'Color', groupColors(g,:), ...
        'LineStyle', styles{g}, 'LineWidth', 1.6, 'Marker', markers{g}, ...
        'MarkerSize', 3.2, 'MarkerIndices', (8+8*g):60:numel(run.t));
    plotHeld(axC, run, run.command, groupColors(g,:), styles{g});
    demand = max(abs(run.raw), [], 2)/run.cfg.U;
    plotHeld(axU, run, demand, groupColors(g,:), styles{g});
    qAll = [qAll; run.x(:,2)]; %#ok<AGROW>
    cAll = [cAll; run.command]; %#ok<AGROW>
    demandMaximum = max(demandMaximum, max(demand));
end
ylabel(axQ, 'Internal angle q (rad)');
ylabel(axC, 'Command c (normalized / s)');
ylabel(axU, {'Raw actuator demand', 'max_i |u_i^{raw}| / U'});
ylim(axQ, paddedBounds(qAll, 0.18));
ylim(axC, paddedBounds([cAll; 0], 0.10));
ylim(axU, [0, 1.12*demandMaximum]);
yline(axU, 1, '-.', 'Clipping limit', 'Color', 'black', ...
    'LabelHorizontalAlignment', 'right', 'FontSize', 8, 'HandleVisibility', 'off');
panelTitle(axQ, sprintf('b  Case %d: internal response', caseIds(1)));
panelTitle(axC, sprintf('c  Case %d: executed command', caseIds(1)));
panelTitle(axU, sprintf('d  Case %d: raw input demand', caseIds(1)));
taskLegend = legend(groupHandles, groupLabels, 'Orientation', 'horizontal', ...
    'NumColumns', 2, 'Box', 'off', 'FontSize', 9, ...
    'TextColor', 'black', 'Color', 'white');
taskLegend.Layout.Tile = 'south';

% Native MATLAB exports. Figures are left open for interactive inspection.
forceLightObjects(fig1);
forceLightObjects(fig2);
drawnow;
artifacts = struct('fixedCommandPNG', fullfile(publicDir, 'fixed-command-probes.png'), ...
    'evaluationPNG', fullfile(publicDir, 'held-out-diagnostics.png'), ...
    'fixedCommandFIG', fullfile(cacheDir, 'fixed-command-probes.fig'), ...
    'evaluationFIG', fullfile(cacheDir, 'held-out-diagnostics.fig'), ...
    'profile', profile);
exportgraphics(fig1, artifacts.fixedCommandPNG, 'Resolution', 300, 'BackgroundColor', 'white');
exportgraphics(fig2, artifacts.evaluationPNG, 'Resolution', 300, 'BackgroundColor', 'white');
savefig(fig1, artifacts.fixedCommandFIG);
savefig(fig2, artifacts.evaluationFIG);
end

function fig = newFigure(name, sizeInches)
fig = figure('Name', name, 'NumberTitle', 'off', 'Color', 'white', ...
    'Units', 'inches', 'Position', [0.3, 0.5, sizeInches], ...
    'DefaultAxesFontName', 'Arial', 'DefaultTextFontName', 'Arial', ...
    'DefaultAxesFontSize', 9, 'DefaultTextFontSize', 9, ...
    'DefaultAxesColor', 'white', 'DefaultAxesXColor', 'black', ...
    'DefaultAxesYColor', 'black', 'DefaultTextColor', 'black');
end

function styleAxes(ax)
hold(ax, 'on');
set(ax, 'Box', 'off', 'LineWidth', 0.8, 'TickDir', 'out', ...
    'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.12, ...
    'GridColor', 'black', 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
    'FontName', 'Arial', 'FontSize', 9, 'Layer', 'top');
end

function panelTitle(ax, value)
h = title(ax, value, 'FontSize', 10, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Color', 'black');
h.Units = 'normalized';
h.Position(1) = 0;
end

function addHeader(fig, value)
annotation(fig, 'textbox', [0.04, 0.935, 0.92, 0.045], 'String', value, ...
    'EdgeColor', 'none', 'Color', 'black', 'BackgroundColor', 'white', ...
    'FontName', 'Arial', 'FontSize', 13, 'FontWeight', 'normal', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

function addCaption(fig, value)
annotation(fig, 'textbox', [0.055, 0.008, 0.89, 0.055], 'String', value, ...
    'EdgeColor', 'none', 'Color', 'black', 'BackgroundColor', 'white', ...
    'FontName', 'Arial', 'FontSize', 9, 'FontWeight', 'normal', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Interpreter', 'tex');
end

function forceLightObjects(fig)
% Only objects owned by these figures are changed; user/root theme is intact.
set(fig, 'Color', 'white');
set(findall(fig, 'Type', 'axes'), 'Color', 'white', ...
    'XColor', 'black', 'YColor', 'black', 'ZColor', 'black', 'GridColor', 'black');
set(findall(fig, 'Type', 'text'), 'Color', 'black');
set(findall(fig, 'Type', 'legend'), 'TextColor', 'black', 'Color', 'white');
end

function plotHeld(ax, run, values, color, style)
% Append the held last value only for displaying the final executed interval.
stairs(ax, run.t, [values(:); values(end)], 'Color', color, ...
    'LineStyle', style, 'LineWidth', 1.6);
end

function bounds = paddedBounds(values, fraction)
limits = [min(values), max(values)];
span = limits(2)-limits(1);
if span == 0, span = max(0.1, abs(limits(1))); end
bounds = limits + [-1, 1]*fraction*span;
end
