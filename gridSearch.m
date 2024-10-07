% Set parameter ranges
wScaleValues = 0.1:0.1:3;   % Range for w_scale
qScaleValues = 0.1:0.1:3;   % Range for q_scale
hScaleValues = 0.1:0.1:3;   % Range for h_scale

% Initialize matrices
[wGrid, qGrid, hGrid] = ndgrid(wScaleValues, qScaleValues, hScaleValues);

% Initialize variables
scores_par = zeros(numel(wGrid), 1);
params_par = zeros(numel(wGrid), 3);
parfor idx = 1:numel(wGrid)
    seed = idx+randi(1000);
    rng(seed);
    % Extract the current parameter combination
    wScale = wGrid(idx);
    qScale = qGrid(idx);
    hScale = hGrid(idx);
    % Run the model w/current parameter set
    [msScore, ~] = bogaczModel(wScale, qScale, hScale, seed);
    scores_par(idx) = msScore;
    params_par(idx, :) = [wScale, qScale, hScale];
end

% Find the best score and corresponding parameters
[bestScore, bestIdx] = max(scores_par);
bestParams = params_par(bestIdx, :);

% Store scores in the grid
scoreGrid = reshape(scores_par, size(wGrid));
[best_w_idx, best_q_idx, best_h_idx] = ind2sub(size(wGrid), bestIdx);


%% Visualization and Saving Figures% Display optimal parameters and best score
disp('Optimal Parameters:');
disp(bestParams);
disp(['Best Score: ', num2str(bestScore)]);

%% Scatter plot of all parameter combinations
figure;
scatter3(params_par(:, 1), params_par(:, 2), params_par(:, 3), 3, scores_par, 'filled');
xlabel('w\_scale');
ylabel('q\_scale');
zlabel('h\_scale');
title('Grid Search Results');
colorbar;
colormap(jet);
grid on;

% Save the scatter plot as a .png file
filename = 'GridSearchScatterPlot.png';
saveas(gcf, filename);
close(gcf);

% Save the entire score grid as a .mat file for later use
save('scoreGrid_results.mat', 'wGrid', 'qGrid', 'hGrid', 'scoreGrid');
