% generates rows of 1-dimensional column vectors
% of training samples for a simple model
function [x, y] = generate()
	x = [rand(2, 50); ones(1,50)];
	y = [0.5 0.1; 0.1 -0.3] * x(1:2,:) + 0.01 * randn(2, 50) + 2;
end