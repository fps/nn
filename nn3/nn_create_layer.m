function layer = nn_create_layer(a, da, number_of_neurons, number_of_neurons_in_previous_layer)
    row = 1;
    
    %activation function
    layer{1, row++} = a;
    
    % derivative of activation function
    layer{1, row++} = da;
    
    % weights
    layer{1, row++} = zeros(number_of_neurons, number_of_neurons_in_previous_layer);
   
    % activations
    layer{1, row++} = zeros(number_of_neurons, 1);
    
    % backprop weights
    layer{1, row++} = zeros(number_of_neurons_in_previous_layer, number_of_neurons);
    
    % errors
    layer{1, row++} = zeros(number_of_neurons, 1);
end