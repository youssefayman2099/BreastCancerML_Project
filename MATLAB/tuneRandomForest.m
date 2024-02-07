function [bestParams, bestScore] = tuneRandomForest(X_train, y_train)
    % Define the parameter grid
    n_estimators_set = [20, 50, 100, 150, 200];
    criterion_set = {'gdi', 'deviance'};  % MATLAB uses 'deviance' instead of 'entropy'
    bootstrap_set = [true, false];

    % Initialize variables to store the best results
    bestScore = 0;
    bestParams = struct('n_estimators', [], 'criterion', [], 'bootstrap', []);

     % Iterate over all combinations in the grid
    for n_estimators = n_estimators_set
        for criterion = criterion_set
            for bootstrap = bootstrap_set
                options = {'Method', 'classification', ...
                           'OOBPrediction', 'On', ...
                           'CategoricalPredictors', 'all', ...
                           'NumPredictorsToSample', 'all', ... % 'all' uses sqrt(number of predictors)
                           'SplitCriterion', criterion{1}};
                if bootstrap
                    options(end+1:end+2) = {'InBagFraction', 1};
                end
                
                % Train the Random Forest model
                rfModel = TreeBagger(n_estimators, X_train, y_train, options{:});

                % Compute the out-of-bag classification error
                ooberror = oobError(rfModel);
                score = 1 - ooberror(end);  % Accuracy as 1 - last OOB Error


                % Update best score and parameters if current model is better
                if score > bestScore
                    bestScore = score;
                    bestParams.n_estimators = n_estimators;
                    bestParams.criterion = criterion{1};
                    bestParams.bootstrap = bootstrap;
                end
            end
        end
    end
    
  
    % Display the best parameters and score
    fprintf('The best parameters for the classifier are:\n');
    disp(bestParams);
    fprintf('The best out-of-bag score is: %.3f\n', bestScore);
end
