function [x, y] = generate()
	x = rand(1, 50);
	y = 0.5 * x + 0.01 * randn(1, 50);
end