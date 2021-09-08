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
CVector = [0.01 0.01 0.1 0.1 1 3 10];
sigmaVector = [0.01 0.01 0.1 0.1 1 3 10];
result = [];

minimum = [0 0 0];

for c = 1:length(CVector)
    for s = 1:length(sigmaVector)
        model = svmTrain(X, y, CVector(c), @(x1, x2) gaussianKernel(x1, x2, sigmaVector(s)));
        predictions = svmPredict(model, Xval);
        result = [ result; mean(double(predictions ~= yval)) CVector(c) sigmaVector(s) ];
    end
end

minimum = sortrows(result)(1,:);
C = minimum(2);
sigma = minimum(3);

% =========================================================================

end
