
function [nn rmses] = test_relu()
    % activation functions and their derivatives
    th = @(x) tanh(x);
    dth = @(x) 1 - tanh(x).**2;

    id = @(x) x;
    did = @(x) ones(size(x));
    
    relu = @(x) max(0, x);
    drelu = @(x) (x > 0);
    %drelu = @(x) (x > 0) + 0.1 * randn(1,1);
    
    mrelu = @(x) max((0.1 * x), x);
    dmrelu = @(x) 0.1 * randn(1,1) + 0.2 + (x > 0) * 0.9;
    
    f = relu;
    df = drelu;
    
    'create the neural net'
    nn = nn_new();

    number_of_hidden_neurons = 50;
    
    nn = nn_add_layer(nn, nn_create_layer(f, df, number_of_hidden_neurons, 1));

    for hidden_layer = 1:5
        nn = nn_add_layer(nn, nn_create_layer(f, df   , number_of_hidden_neurons, number_of_hidden_neurons));
    end
    
    nn = nn_add_layer(nn, nn_create_layer(id, did, 1, number_of_hidden_neurons));

    nn = nn_initialize_forward_weights_gaussian(nn, 2);

    'test data to learn'
    number_of_samples = 1000;
    x = rand(1, number_of_samples) * 2 - 1;
    y = 1 * sin(5 * x);
    
    minibatch_size = 40;
    
    'training'
    number_of_epochs = 1000;
    
    nn = nn_initialize_backward_weights_uniform(nn, 1);
    %nn = nn_initialize_backward_weights_gaussian(nn, 0.1);
    %nn = nn_normalize_backward_weights(nn, 2);
    %nn = nn_update_backward_weights_transpose(nn);
    
    rmses = [];
    
    for epoch = 1:number_of_epochs
        epoch
        p = randperm(number_of_samples)(1:minibatch_size);
        
        % forward pass
        'forward pass'
        tic
        nn = nn_forward_pass(nn, x(:,p));
        toc
        
        nn_assert_consistency(nn);
        
        rmse = sqrt(sum((y(:,p) - nn{rows(nn)}.activations).^2) / number_of_samples)
        
        plot(x(:,p), nn{rows(nn)}.activations, '.', x(:,p), y(:,p), '+'); sleep(0.01); 
        
        rmses = [rmses rmse];
        
        % update backwards weights
        'update backwards weights'
        %nn = nn_update_backward_weights_gaussian(nn, 0.001);
        nn = nn_update_backward_weights_transpose(nn);
        nn_assert_consistency(nn);
        
        % update weights 
        'backwards pass'
        tic
        nn = nn_backward_pass(nn, y(:,p));
        toc
        
        nn_assert_consistency(nn);        
        
        tic
        nn = nn_sgd(nn, 0.02);
        toc
        
        nn_assert_consistency(nn);
        
        %nn_count_saturated_relus(nn);
        nn = nn_reinitialize_saturated_relus(nn, 0.00001);
    end
end