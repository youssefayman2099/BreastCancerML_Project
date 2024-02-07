% Create function for performance metrics
function [Accuracy, Sensitivity, Specificity, Precision, F_measure] = metrics(X, y_test, trainedClassifier)
    % predictions are made using the trained classifier
    y_pred = predict(trainedClassifier, X);

    % create an if condition to ensure y_pred is of the same type as y_test
    if iscell(y_test) && isnumeric(y_pred)
        y_pred = cellstr(num2str(y_pred));
    elseif isnumeric(y_test) && iscell(y_pred)
        y_pred = str2double(y_pred);
    end

    % Create the confusion matrix
    cm = confusionmat(y_test, y_pred);

    % Extract true negatives, true positives, false negatives, and false positives
    tn = cm(1,1);
    fp = cm(1,2);
    fn = cm(2,1);
    tp = cm(2,2);

    % Calculations for all 5 performance metrics
    Accuracy = (tp + tn) / (tp + tn + fp + fn);
    Sensitivity = tp / (tp + fn);  % Recall
    Specificity = tn / (tn + fp);
    Precision = tp / (tp + fp);
    F_measure = 2 * tp / (2 * tp + fp + fn);

    % Display the metrics to show how the model performed overall
    fprintf('Accuracy=%.3f\n', Accuracy);
    fprintf('Sensitivity=%.3f\n', Sensitivity);
    fprintf('Specificity=%.3f\n', Specificity);
    fprintf('Precision=%.3f\n', Precision);
    fprintf('F-measure=%.3f\n', F_measure);
    % Display the confusion matrix chart on each model
    confusionMatrixChart = confusionchart(y_test, y_pred);
    confusionMatrixChart.Title = 'Confusion Matrix';
    confusionMatrixChart.XLabel = 'Predicted Label';
    confusionMatrixChart.YLabel = 'True Label';
end