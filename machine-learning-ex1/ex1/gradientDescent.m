function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
fprintf("theta1 = %f\n", theta(1));
fprintf("theta2 = %f\n", theta(2));
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
     %% fprintf("computeCost:= %f\n", computeCost(X,y,theta));
   %   thetaTemp1 = theta(1)-(alpha)*computeCost(X,y,theta);
   %   thetaTemp2 = theta(2)-(alpha)*computeCost(X,y,theta);
   
      predictions = X*theta;
      xVector = X(:,[1]);
      sqrErrors = (predictions-y).*xVector;
      thetaTemp1 = theta(1)-(alpha*1/(m)*sum(sqrErrors));
      
      %theta 2
      predictions = X*theta;
      xVector = X(:,[2]);
      sqrErrors = (predictions-y).*xVector;
      thetaTemp2 = theta(2)-(alpha*1/(m)*sum(sqrErrors));
      theta(1) = thetaTemp1;
      theta(2) = thetaTemp2;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
