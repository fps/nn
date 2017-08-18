
function nn = test()
    % activation functions and their derivatives
    th = @(x) tanh(x);
    dth = @(x) 1 - tanh(x).**2;

    identity = @(x) x;
    derivative_of_identity = @(x) ones(size(x));
    
    relu = @(x) max(-0.1, 0.9 * x);
    drelu = @(x) (x > -0.1) - 0.1;
    
    'create the neural net'
    nn = nn_new();

    number_of_hidden_neurons = 100;
    
    nn = nn_add_layer(nn, nn_create_layer(th, dth, number_of_hidden_neurons, 1));

    for hidden_layer = 1:2
        nn = nn_add_layer(nn, nn_create_layer(th, dth   , number_of_hidden_neurons, number_of_hidden_neurons));
    end
    
    nn = nn_add_layer(nn, nn_create_layer(identity, derivative_of_identity, 1, number_of_hidden_neurons));

    nn = nn_initialize_forward_weights_gaussian(nn, 1);

    'test data to learn'
    number_of_samples = 100;
    x = rand(1, number_of_samples) * 2 - 1;
    y = 1 * sin(5 * x);
    
    'training'
    number_of_epochs = 5000;
    
    %nn = nn_initialize_backward_weights_uniform(nn, 0.5);
    %nn = nn_initialize_backward_weights_gaussian(nn, 0.5);
    nn = nn_normalize_backward_weights(nn, 2);
    
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
        
        plot(x(:,p), nn{rows(nn)}.activations, '.', x(:,p), y(:,p), '*'); sleep(0.01); 
        
        % update backwards weights
        'update backwards weights'
        %nn = nn_update_backward_weights_gaussian(nn, 0.001);
        nn = nn_update_backward_weights_transpose(nn);
        nn_assert_consistency(nn);
        
        % update weights 
        'backwards pass'
        tic
        nn = nn_backward_pass(nn, y(:,p), 0.00001);
        toc
        nn_assert_consistency(nn);        
    end
end