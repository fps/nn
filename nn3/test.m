
function nn = test()

    sigmoid = @(x) tanh(x);
    derivative_of_sigmoid = @(x) 1 - tanh(x)**2;

    identity = @(x) x;
    derivative_of_identity = @(x) 1;

    nn = nn_new()

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 10, 1))

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 10))

    nn = nn_add_layer(nn, nn_create_layer(identity, derivative_of_identity, 1, 20))

    nn = nn_initialize_weights_gaussian(nn)

    x = -2:0.01:2;
    nn = nn_forward_pass(nn, x);
    plot(x, nn{3}{1,4}, '.')
end