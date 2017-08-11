function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

fprintf('Sizex of X: %f\n', size(X))
fprintf('Sizex of y: %f\n', size(y))
fprintf('Sizex of theta: %f\n', size(theta))

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

%    num_feat=size(X,2);

%    for feat_idx = 1:num_feat,
%      local_theta(feat_idx,1) = theta(feat_idx,1) - alpha * (1 / m) * sum(((X * theta) - y).* X(:,feat_idx));
%    end;

%    theta=local_theta;

    theta = theta - (alpha/m) * X' * (X * theta - y);

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end
fprintf('Sizex of theta: %f\n', size(theta))

end
