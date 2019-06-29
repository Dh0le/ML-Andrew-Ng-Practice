function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, numn,/_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
%for here we add bias theta for bias unit as well size of Theta1 = 5000 x 401
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%for here we add bias theta for bias unit as well size of Theta2 = 10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% FORWARD PROPOGATION
a1 = [ones(m,1) X]; %add bias unit into input layer a1 size = 5000 x401
z2 = (a1*Theta1'); %calculate the z value       Theta1 size = 5000 x25
a2 = [ones(size(z2,1),1) sigmoid(z2)]; %add bias unit into hidden layer. a2size = [5000 x 26]
z3 = a2*Theta2';  %a2 size = 5000 x 26 z3  size of Theta2 = 10 x 26 z3 size = 5000 x 10
h = sigmoid(z3); %predict the hypothesis value.
a3 = h;
I = eye(num_labels);
Y = zeros(m,num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end
%size of Y = 5000 x1 10
J = (sum(sum(-Y.*log(h))) - sum(sum((1-Y).*(log(1-h)))))/m;

% REGULARIZATION
regularization_term = (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))) + sum(sum(Theta2(:,2:end).^2))); %remember to exclude the first theta value
%Because no bias unit should be penalized. The Theta1 size here = 25 x 400 Theta2 as well
J = J + regularization_term;
%Here is the vectorized implementation 
% BACK PROPOGATION
delta3 = a3 - Y;  % has same dimensions as a3 [5000 x 10]
delta2 = (delta3*Theta2).*[ones(size(z2,1),1) sigmoidGradient(z2)];     % has same dimensions as a2 add back the bias unit for  calculation
%size of delta2 = 5000 x 26
D1 = delta2(:,2:end)' * a1;    % has same dimensions as Theta1 [5000 x 25]' x [5000 x 401] = [25 x 401]
D2 = delta3' * a2;    % has same dimensions as Theta2 [5000 x 10]' x [26 x 10] = [10 x 26]

Theta1_grad = Theta1_grad + (1/m) * D1;
Theta2_grad = Theta2_grad + (1/m) * D2;


% REGULARIZATION OF THE GRADIENT

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));  %remember to exclude the bias unit from the penalization
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));  %that is the j = 0 case
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
