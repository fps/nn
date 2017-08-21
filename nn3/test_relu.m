
function nn = test_relu()
    % activation functions and their derivatives
    th = @(x) tanh(x);
    dth = @(x) 1 - tanh(x).**2;

    id = @(x) x;
    did = @(x) ones(size(x));
    
    relu = @(x) max(0, x);
    drelu = @(x) (x > 0);
    
    'create the neural net'
    nn = nn_new();

    number_of_hidden_neurons = 10;
    
    nn = nn_add_layer(nn, nn_create_layer(relu, drelu, number_of_hidden_neurons, 1));

    for hidden_layer = 1:5
        nn = nn_add_layer(nn, nn_create_layer(relu, drelu   , number_of_hidden_neurons, number_of_hidden_neurons));
    end
    
    nn = nn_add_layer(nn, nn_create_layer(id, did, 1, number_of_hidden_neurons));

    nn = nn_initialize_forward_weights_gaussian(nn, 3);

    'test data to learn'
    number_of_samples = 200;
    x = rand(1, number_of_samples) * 2 - 1;
    y = 1 * sin(5 * x);
    
    'training'
    number_of_epochs = 5000;
    
    %nn = nn_initialize_backward_weights_uniform(nn, 0.5);
    %nn = nn_initialize_backward_weights_gaussian(nn, 0.1);
    %nn = nn_normalize_backward_weights(nn, 2);
    nn = nn_update_backward_weights_transpose(nn);
    
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
        
        plot(x(:,p), nn{rows(nn)}.activations, '.', x(:,p), y(:,p), '+'); sleep(0.01); 
        
        % update backwards weights
        'update backwards weights'
        %nn = nn_update_backward_weights_gaussian(nn, 0.001);
        nn = nn_update_backward_weights_transpose(nn);
        nn_assert_consistency(nn);
        
        % update weights 
        'backwards pass'
        tic
        nn = nn_backward_pass(nn, y(:,p), 0.0002);
        toc
        nn_assert_consistency(nn);        
        
        %nn_count_saturated_relus(nn);
        nn = nn_reinitialize_saturated_relus(nn, 0.0001);
    end
end