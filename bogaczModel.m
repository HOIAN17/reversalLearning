function [ms_score, astr] = bogaczModel(w_scale, q_scale, h_scale, seed, plotTrue)
    % Simulates an operant conditioning experiment with multiple reversals
    % This function models an agent learning to choose actions based on stimuli,
    % with rewards given for correct stimulus-action pairings. The correct actions
    % can change at multiple reversal points during the experiment.
    if nargin <5
        plotTrue =false;
        if nargin < 4
            seed = rand(1);
            sprintf('Seed set %.2f',seed)     
        end
    end
    sprintf('w: %d q: %d h: %d',w_scale,q_scale,h_scale);

    %% Scaling factors
    w_scale = w_scale;
    q_scale = q_scale;
    h_scale = h_scale;

    %% Constants for trials

    TRIALS = 400;               % Total number of trials
    REVERSALS = [100,200,300];  % Array of reversal points

    %% Timing parameters
    trialDuration = 2;  % Duration of a single trial (seconds)
    dt = 0.001;         % Time step (seconds)
    tau = 0.05;         % Time constant of the integrator (seconds)
    taudelta = 0.02;    % Time constant for delta variables (seconds)
    CStime = 1;         % Time when the stimulus (CS) is presented (seconds)
    delay = 0.1;        % Delay for critic's prediction error calculation (seconds)

    %% Learning rates
    alphag = 0.1;       % Learning rate for the goal-directed system
    alphacplus = 0.5;   % Learning rate for the critic (positive prediction error)
    alphacminus = 0.1;  % Learning rate for the critic (negative prediction error)
    alphah = 0.05;      % Learning rate for the habit system

    %% Other parameters
    exploration = 1;    % Standard deviation of noise added to the action; originally 2
    reward_noise = 0.5; % Standard deviation of noise added to reward    ;
    alphaprec = 0.03;   % Precision learning rate
    COMPETITION = 1;    % Strength of competition between actions

    %% Initialize parameters for critic, actor, and habit systems
    w = zeros(2, TRIALS + 1);         % Weights for the critic
    w(:, 1) = 0.1;
    q = zeros(2, 2, TRIALS + 1);      % Weights for the goal-directed system
    q(:, :, 1) = 0.1;
    h = zeros(2, 2, TRIALS + 1);      % Weights for the habit system

    %% Precisions (variance estimates)
    varianceh = ones(1, TRIALS + 1);  % Variance for the habit system
    varianceg = ones(1, TRIALS + 1);  % Variance for the goal-directed system
    varianceh(1) = 100;               % Initially the habit system is uncertain
    minvar = 0.2;                     % Minimum value for the variances

    %% Variables for storing prediction errors
    maxdc = zeros(1, TRIALS);         % Max delta for the critic
    maxdg = zeros(1, TRIALS);         % Max delta for the goal-directed system
    maxdh = zeros(1, TRIALS);         % Max delta for the habit system

    %% Variables for storing results
    grade = zeros(1, TRIALS);         % Performance grade (correct choices)
    astr = zeros(2, TRIALS);          % Strength of beliefs/actions

    %% Set random seed for reproducibility
    rng(seed);

    %% Main simulation loop over trials
    for trial = 1:TRIALS
        %% Determine the correct action based on multiple reversals
        stimulus = randi(2);  % Randomly select stimulus 1 or 2

        % Determine the number of reversals occurred up to the current trial
        num_reversals = sum(trial >= REVERSALS);

        % Decide the correct action based on the current reversal state
        if num_reversals == 0  % Before the first reversal
            correct = stimulus;
        elseif mod(num_reversals, 2) == 1
            correct = 3 - stimulus;  % Reverse the action mapping
        else
            correct = stimulus;      % Switch back to original mapping
        end

        %% Stimulus initialization
        s = zeros(2, trialDuration/dt + 1);
        s(stimulus, round(CStime/dt):end) = 1;  % Present the stimulus from CStime onwards

        %% Initialize variables for the trial
        v = zeros(1, trialDuration/dt + 1);        % Value estimated by the critic
        deltac = zeros(1, trialDuration/dt);       % Prediction error for the critic
        a = zeros(2, trialDuration/dt + 1);        % Action values
        deltag = zeros(1, trialDuration/dt);       % Prediction error for the goal-directed system
        deltah = zeros(1, trialDuration/dt);       % Prediction error for the habit system

        %% Time loop within the trial
        for i = 1:trialDuration/dt
            %% Step 1: Critic value calculation
            v_input = w_scale * w(:, trial)' * s(:, i);
            v(i+1) = v(i) + dt/tau * (v_input - v(i));

            % Calculate the delayed value for prediction error
            if i - round(delay/dt) < 1
                oldv = v(1);
            else
                oldv = v(i - round(delay/dt));
            end

            %% Step 2: Critic prediction error (delta_c)
            deltac(i+1) = deltac(i) + dt/taudelta * (v(i) - oldv - deltac(i));

            %% Step 3: Goal-directed prediction error (delta_g)
            q_input = a(:, i)' * q(:, :, trial) * q_scale * s(:, i);
            deltag(i+1) = deltag(i) + dt/taudelta * ((v(i) - q_input) - varianceg(trial) * deltag(i));

            %% Step 4: Update actions
            % Mutual inhibition between actions
            mut_inh = COMPETITION * [a(2, i); a(1, i)];

            % Inputs from goal-directed and habit systems
            goal_input = deltag(i) * q(:, :, trial) * s(:, i);
            habit_input = (h(:, :, trial) * s(:, i)) / varianceh(trial);

            % Update action values
            a(:, i+1) = a(:, i) + dt/tau * (goal_input + habit_input - a(:, i) / varianceh(trial) - mut_inh);

            % Bound actions between 0 and 1
            a(:, i+1) = min(max(a(:, i+1), 0), 1);

            %% Step 5: Habit prediction error (delta_h)
            h_input = a(:, i)' * h(:, :, trial) * h_scale * s(:, i);
            deltah(i+1) = deltah(i) + dt/taudelta * (sum(a(:, i)) - h_input - deltah(i));
        end

        %% Determine the chosen action at the end of the trial
        [~, choice] = max(a(:, end) + exploration * randn(2, 1));

        %% Calculate the reward based on correctness
        r = (choice == correct) + reward_noise * randn;

        %% Update performance grade and belief strength
        grade(trial) = (choice == correct);

        % Store the action strengths (beliefs)
        if correct == 1
            astr(:, trial) = a(:, end);
        else
            astr(:, trial) = a([2, 1], end);  % Swap actions if correct action is 2
        end

        %% Residual prediction errors at the end of the trial
        deltac_r = r - v(end);  % Critic prediction error with actual reward

        % Action vector for chosen action
        action_vector = zeros(2, 1);
        action_vector(choice) = 1;

        % Goal-directed prediction error with actual reward
        deltag_r = (r - action_vector' * q(:, :, trial) * q_scale * s(:, end)) / varianceg(trial);

        % Habit prediction error with actual action
        deltah_r = sum(action_vector) - action_vector' * h(:, :, trial) * h_scale * s(:, end);

        %% Update weights for the critic
        if deltac_r > 0
            w(:, trial + 1) = w(:, trial) + alphacplus * deltac_r * s(:, end);
        else
            w(:, trial + 1) = w(:, trial) + alphacminus * deltac_r * s(:, end);
        end

        %% Update weights for the goal-directed system
        q(:, :, trial + 1) = q(:, :, trial) + alphag * deltag_r * action_vector * s(:, end)';

        %% Update weights for the habit system
        h(:, :, trial + 1) = h(:, :, trial) + alphah * deltah_r * (action_vector - 0.5) * s(:, end)';
        h(:, :, trial + 1) = max(h(:, :, trial + 1), 0);  % Ensure non-negative synaptic weights

        %% Update variances (precisions)
        varianceh(trial + 1) = varianceh(trial) + alphah * (deltah_r^2 - varianceh(trial));
        varianceh(trial + 1) = max(varianceh(trial + 1), minvar);

        varianceg(trial + 1) = varianceg(trial) + alphaprec * ((deltag_r * varianceg(trial))^2 - varianceg(trial));
        varianceg(trial + 1) = max(varianceg(trial + 1), minvar);

        %% Store maximum prediction errors for plotting
        maxdc(trial) = max(abs(deltac));
        maxdg(trial) = max(abs(deltag));
        maxdh(trial) = max(abs(deltah));
    end

    %% Calculate the mean score based on belief strength after the reversals
    ms_score = mean(astr(1, :));
    if plotTrue ==true
        %% Plotting results
        % Figure 1: Variance estimates over trials
        figure;
        subplot(2, 1, 1);
        plot(varianceg, 'Color', [1, 0.5, 0], 'LineWidth', 1.5);  % Goal-directed variance
        hold on;
        plot(varianceh, 'b', 'LineWidth', 1.5);                   % Habit variance
        xlabel('Trial');
        ylabel('Variance Estimate');
        legend('\Sigma_g', '\Sigma_h', 'Location', 'Best');
        title('Variance Estimates over Trials');
        xlim([0 TRIALS]);
        grid on;
        % Mark reversal points
        for rev = REVERSALS
            xline(rev, '--k', 'LineWidth', 1);
        end
    
        % Figure 2: Prediction errors over trials
        subplot(2, 1, 2);
        plot(maxdc, 'r', 'LineWidth', 1);              % Critic prediction error
        hold on;
        plot(maxdg, 'Color', [1, 0.5, 0], 'LineWidth', 1);  % Goal-directed prediction error
        plot(maxdh, 'b', 'LineWidth', 1);              % Habit prediction error
        xlabel('Trial');
        ylabel('Peak Prediction Error');
        legend('\delta_v', '\delta_g', '\delta_h', 'Location', 'Best');
        title('Prediction Errors over Trials');
        xlim([0, TRIALS]);
        grid on;
        % Mark reversal points
        for rev = REVERSALS
            xline(rev, '--k', 'LineWidth', 1);
        end
    
        % Figure 3: Synaptic weights over trials
        figure;
        subplot(2, 1, 1);
        % Compute average weights for concordant (pro) and discordant (anti) tasks
        q_pro = squeeze((q(1, 1, :) + q(2, 2, :))) * q_scale / 2;
        h_pro = squeeze((h(1, 1, :) + h(2, 2, :))) * h_scale / 2;
        q_anti = squeeze((q(1, 2, :) + q(2, 1, :))) * q_scale / 2;
        h_anti = squeeze((h(1, 2, :) + h(2, 1, :))) * h_scale / 2;
    
        plot(q_pro, 'Color', [1, 0.5, 0], 'LineWidth', 1.5);  % Goal-directed PRO weights
        hold on;
        plot(h_pro, 'b', 'LineWidth', 1.5);                   % Habit PRO weights
        plot(q_anti, 'Color', [1, 0.5, 0], 'LineStyle', '--', 'LineWidth', 1.5);  % Goal-directed ANTI weights
        plot(h_anti, 'b', 'LineStyle', '--', 'LineWidth', 1.5);                   % Habit ANTI weights
        xlabel('Trial');
        ylabel('Synaptic Weights');
        legend('Goal_{PRO}', 'Habit_{PRO}', 'Goal_{ANTI}', 'Habit_{ANTI}', 'Location', 'Best');
        title('Synaptic Weights for Concordant and Discordant Tasks');
        xlim([0, TRIALS]);
        grid on;
        % Mark reversal points
        for rev = REVERSALS
            xline(rev, '--k', 'LineWidth', 1);
        end
    
        % Figure 4: Belief strength over trials
        subplot(2, 1, 2);
        plot(astr(1, :), 'k', 'LineWidth', 1.5);
        xlabel('Trial');
        ylabel('Belief Strength');
        title('Belief Strength across Trials');
        xlim([0, TRIALS]);
        ylim([0, 1.1]);
        grid on;
        % Mark reversal points
        for rev = REVERSALS
            xline(rev, '--k', 'LineWidth', 1);
        end
    end
end
