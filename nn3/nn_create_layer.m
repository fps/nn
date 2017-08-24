function layer = nn_create_layer(activation_function, derivative_of_activation_function, number_of_neurons, number_of_neurons_in_previous_layer)
    layer = struct();
    
    layer.f = activation_function;
    
    layer.df = derivative_of_activation_function;
    
    layer.number_of_neurons = number_of_neurons;
    
    layer.number_of_neurons_in_previous_layer = number_of_neurons_in_previous_layer;
    
    layer.forward_weights = zeros(number_of_neurons, number_of_neurons_in_previous_layer + 1);
    
    layer.gradients = zeros(size(layer.forward_weights));
    
    layer.backward_weights = layer.forward_weights';
    
    layer.inputs = zeros(number_of_neurons_in_previous_layer + 1, 1);
    
    layer.activations = zeros(number_of_neurons, 1);
    
    layer.sums = zeros(number_of_neurons, 1);
    
    layer.errors = zeros(number_of_neurons, 1);
end