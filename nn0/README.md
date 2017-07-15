# nn0

Given a "neural net" which is just a single linear neuron (even without bias terms in the summation):

 [1] <pre>f(w, x) = y = w x<pre>,

x \in |R, w \in |R, so f:|R -> |R and given a sequence of training samples, X = x_1, ..., x_n, Y = y_1, ..., y_n, we can define the loss function 

L(w, X, Y) = 1/2 sum_i |y_i - f(w, x_i)|^2 [2].

Note that this is a sum, so we can also define "individual" loss functions per sample:

L_i(w, x_i, y_i) = 1/2 |y_i - f(w, x_i)|^2 [3].

To find the gradient of the loss function we differentiate L_i with respect to w:

dL_i / dw = -1 * (y_i - f(w, x_i)) * x_i [4]

(chain rule: dg(h(x)) / dx = (dg / dh) * (dh / dx), in this case g = y^2, and h = y_i - f(w, x_i)) = y_i - w * x_i.

Since the derivative of a sum of functions is the sum of the derivatives we can immediately also write the "batch" gradient:

dL / dw = sum_i -1 * (y_i - f(w, x_i)) * x_i [5].

The term 

e_i(w) = y_i - f(w, x_i) = y_i - w * x_i [6]

we call the error-term which depends on the current w which is the per-sample-error. So we can rewrite dL/dw and dL_i/dw for brevity as

dL_i/dw = -e_i * x_i,

dL/dw = sum_i -e_i * x_i.

Now we choose a learning rate r << 1 and update the weights in the direction of greatest _descent_,

delta w = r * -dL_i/dw = e_i * x_i [7]

for single sample updates or 

delta w = r * -dL/dw = sum_i e_i * x_i [8] 

for batch updates. Now we're ready to cast our toy model into octave code. See

[model.m](model.m) - our very simple model f(w,x) = w  x

[generate.m](generate.m) - generating some random samples with noise

[optimize_lame.m](optimize_lame.m) - optimizing in a super unoptimized way

[test_lame.m](test_lame.m) - a little test program with animated plots :)

[optimize.m](optimize.m) - optimizing in a slightly less super unoptimized way

[test.m](test.m) - a little test program with animated plots :)
