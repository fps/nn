# nn0

Given a "neural net" which is just a single linear neuron (even without bias terms in the summation):

 [1] <code>f(w, x) = y = w x, x \in |R, w \in |R, f:|R -> |R</code>,

and given a sequence of training samples, 

<code>X = x_1, ..., x_n, Y = y_1, ..., y_n</code>, 

we can define the loss function 

[2] <code>L(w, X, Y) = 1/2 sum_i |y_i - f(w, x_i)|^2</code>.

Note that this is a sum, so we can also define "individual" loss functions per sample:

[3] <code>L_i(w, x_i, y_i) = 1/2 |y_i - f(w, x_i)|^2</code>.

To find the gradient of the loss function we differentiate L_i with respect to w:

[4] <code>dL_i / dw = -1 * (y_i - f(w, x_i)) * x_i</code>]

using the chain rule: 

<code>dg(h(x)) / dx = (dg / dh) * (dh / dx)</code>, 

in this case 

<code>g = y^2</code>, and <code>h = y_i - f(w, x_i)) = y_i - w * x_i</code>.

Since the derivative of a sum of functions is the sum of the derivatives we can immediately also write the "batch" gradient:

[5] <code>dL / dw = sum_i -1 * (y_i - f(w, x_i)) * x_i</code>.

The term 

[6] <code>e_i(w) = y_i - f(w, x_i) = y_i - w * x_i</code>

we call the error-term (which depends on the current w) which is the per-sample-error. So we can rewrite <code>dL/dw</code> and <code>dL_i/dw</code> for brevity as

<code>dL_i/dw = -e_i * x_i</code>,

<code>dL/dw = sum_i -e_i * x_i</code>.

Now we choose a learning rate r << 1 and update the weights in the direction of greatest _descent_,

[7] <code>delta w = r * -dL_i/dw = e_i * x_i</code>

for single sample updates or 

[8] <code>delta w = r * -dL/dw = sum_i e_i * x_i</code>

for batch updates. Now we're ready to cast our toy model into octave code. See

[model.m](model.m) - our very simple model <code>f(w,x) = w  x</code>

[generate.m](generate.m) - generating some random samples with noise

[optimize_lame.m](optimize_lame.m) - optimizing in a super unoptimized way

[test_lame.m](test_lame.m) - a little test program with animated plots :)

[optimize.m](optimize.m) - optimizing in a slightly less super unoptimized way

[test.m](test.m) - a little test program with animated plots :)
