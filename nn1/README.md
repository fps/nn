# nn1

We extend the simple neural net from nn0 to have n-dimensional inputs and m-dimensional outputs (in nn-lingo: m output neurons). We stick to the linear model:

[1] <code>f(w, x) = y = w x, x \in |R^n, w \in |R^m, f:|R^m -> |R^n</code>,

Our loss function looks still the same on the surface. The only thing that changed is that <code>|  |^2</code> now stands for the squared norm of the vector difference.

[2] <code>L(w, X, Y) = 1/2 sum_i |y_i - f(w, x_i)|^2</code>.

We now write <code>v_ij</code> for the j-th component of the i-th vector. The squared norm of a vector <code>v_i</code> is given by

[3] <code>| v_i |^2 = sum_j v_i^2</code>.

More explicitly written (using [3]) the loss function [2] looks like

[4] <code>L(w, X, Y) = 1/2 sum_i sum_j(y_ij - f(w, x_i)_j)^2</code>.

To find the gradient of the loss function we differentiate L_i with respect to w:

[5] <code>dL / dw_j = -sum_i (y_ij - (w_j * x_i)) * x_i</code>

Now we can present our updated example code for a 2-dim in, 2-dim out nn:

[model.m](model.m) 

[generate.m](generate.m) 

[optimize.m](optimize.m)

[test.m](test.m)
