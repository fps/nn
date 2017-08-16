
function nn = test()
    % activation functions and their derivatives
    sigmoid = @(x) tanh(x);
    derivative_of_sigmoid = @(x) 1 - tanh(x).**2;

    identity = @(x) x;
    derivative_of_identity = @(x) ones(size(x));
    
    % create the neural net
    nn = nn_new();

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 10, 2));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 10));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    %nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    %nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 20, 20));

    %nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 8, 9));

    %nn = nn_add_layer(nn, nn_create_layer(sigmoid, derivative_of_sigmoid, 7, 8));

    nn = nn_add_layer(nn, nn_create_layer(identity, derivative_of_identity, 1, 20));

    nn = nn_initialize_forward_weights_gaussian(nn);

    % test data to learn
    number_of_samples = 100;
    x = [rand(1, number_of_samples) * 2 - 1; ones(1, number_of_samples)];
    y = 1 * sin(5 * x(1,:));
    
    % training
    number_of_epochs = 5000;
    
    nn = nn_initialize_backward_weights_gaussian(nn, 0.5);
    
    for epoch = 1:number_of_epochs
        epoch
        p = randperm(number_of_samples);
        
        % forward pass
        'forward pass'
        tic
        nn = nn_forward_pass(nn, x(:,p));
        toc
        nn_assert_consistency(nn);
        
        rmse = sqrt(sum((y(:,p) - nn{rows(nn)}.activations).^2) / number_of_samples)
        
        plot(x(1,p), nn{rows(nn)}.activations, '.', x(1,p), y(:,p), '*'); sleep(0.01); 
        
        % update backwards weights
        'update backwards weights'
        %nn = nn_update_backward_weights_gaussian(nn, 0.001);
        %nn = nn_update_backward_weights_transpose(nn);
        nn_assert_consistency(nn);
        
        % update weights 
        'backwards pass'
        tic
        nn = nn_backward_pass(nn, y(:,p), 0.0001);
        toc
        nn_assert_consistency(nn);        
    end
end