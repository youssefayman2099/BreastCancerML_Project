function [bestParams, bestScore] = tuneLogisticRegression(X_train, y_train)
    % Define the range of C values
    C_values = [0.01, 0.1, 1, 10, 100];
    numC = length(C_values);

    % Initialize variables to store best results
    bestScore = 0;
    bestParams = struct('C', [], 'Lambda', []);

    % Iterate over C values
    for i = 1:numC
        % Convert C to Lambda (regularization strength in MATLAB)
        lambda = 1 / C_values(i);

        % Train the logistic regression model with cross-validation
        cvmodel = fitclinear(X_train, y_train, 'Learner', 'logistic', 'Lambda', lambda, 'KFold', 5);

        % Compute the mean accuracy
        meanAcc = 1 - kfoldLoss(cvmodel);

        % Update best score and parameters if current model is better
        if meanAcc > bestScore
            bestScore = meanAcc;
            bestParams.C = C_values(i);
            bestParams.Lambda = lambda;
        end
    end

    % Display the best parameters and score
    fprintf('The best parameters for the classifier are:\n');
    disp(bestParams);
    fprintf('The best training score is: %.3f\n', bestScore);
end