
function nn = test()
    % activation functions and their derivatives
    sigmoid = @(x) tanh(x);
    derivative_of_sigmoid = @(x) 1 - tanh(x).**2;

    identity = @(x) x;
    derivative_of_identity = @(x) ones(size(x));
    
    % create the neural net
    nn = nn_new();

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 2));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 25, 20));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 25, 25));

    %nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    %nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    %nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 8, 9));

    %nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 7, 8));

    nn = nn_add_layer(nn, nn_create_layer(identity, derivative_of_identity, 1, 25));

    nn = nn_initialize_weights_gaussian(nn);

    % test data to learn
    number_of_samples = 300;
    x = [rand(1, number_of_samples) * 2 - 1; ones(1, number_of_samples)];
    y = 1 * sin(5 * x(1,:));
    
    % training
    number_of_epochs = 500;
    
    nn = nn_initialize_backward_weights_gaussian(nn);
    
    for epoch = 1:number_of_epochs
        epoch
        p = randperm(number_of_samples);
        
        % forward pass
        'forward pass'
        nn = nn_forward_pass(nn, x(:,p));
        nn_assert_consistency(nn);
        
        rmse = sqrt(sum((y(:,p) - nn{rows(nn)}.activations).^2) / number_of_samples)
        
        plot(x(1,p), nn{rows(nn)}.activations, '.', x(1,p), y(:,p), '*'); sleep(0.01); 
        
        % update backwards weights
        'update backwards weights'
        % nn = nn_update_backward_weights_transpose(nn);
        nn_assert_consistency(nn);
        
        % update weights 
        'backwards pass'
        nn = nn_backward_pass(nn, y(:,p), 0.05);
        nn_assert_consistency(nn);        
    end
end