function layer = nn_create_layer(activation_function, derivative_of_activation_function, number_of_neurons, number_of_neurons_in_previous_layer)
    layer = struct();
    
    layer.f = activation_function;
    layer.df = derivative_of_activation_function;
    layer.number_of_neurons = number_of_neurons;
    layer.number_of_neurons_in_previous_layer = number_of_neurons_in_previous_layer;
    layer.forward_weights = zeros(number_of_neurons, number_of_neurons_in_previous_layer);
    layer.backward_weights = zeros(number_of_neurons_in_previous_layer, number_of_neurons);
    layer.activations = zeros(number_of_neurons, 1);
    layer.errors = zeros(number_of_neurons, 1);
    
    % row = 1;
    
    % activation function
    % layer{1, row++} = a;
    
    % derivative of activation function
    % layer{1, row++} = da;
    
    % weights
    % layer{1, row++} = zeros(number_of_neurons, number_of_neurons_in_previous_layer);
   
    % activations
    % layer{1, row++} = zeros(number_of_neurons, 1);
    
    % backprop weights
    % layer{1, row++} = zeros(number_of_neurons_in_previous_layer, number_of_neurons);
    
    % errors
    % layer{1, row++} = zeros(number_of_neurons, 1);
end