function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_params = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_params = [0.01 0.03 0.1 0.3 1 3 10 30]';
prediction_error = zeros(length(C_params), length(sigma_params));
for c = 1:length(C_params)
  for s = 1:length(sigma_params)
    C_param = C_params(c);
    sigma_param = sigma_params(s);
    training_model = svmTrain(X, y, C_param, @(x1, x2) gaussianKernel(x1, x2, sigma_param));
    predictions = svmPredict(training_model, Xval);
    prediction_error(c,s) = mean(double(predictions ~= yval));
  endfor
end
% =========================================================================
[X_values, row_value] = min(prediction_error);
[~, min_x_val] = min(X_values);
min_val_row = row_value(min_x_val);

C = C_params(min_val_row);
sigma = sigma_params(min_x_val);

end
