 
function nn_assert_consistency(nn)
    for layer = 1:rows(nn)
        % assert(columns(nn{layer}.activations) == columns(nn{layer}.errors));
        
        assert(rows(nn{layer}.activations) == nn{layer}.number_of_neurons);
        
        assert(rows(nn{layer}.errors) == nn{layer}.number_of_neurons);
    
        assert(nn{layer}.number_of_neurons_in_previous_layer + 1 == columns(nn{layer}.forward_weights), ['in layer: ' int2str(layer)]);
        
        assert(nn{layer}.number_of_neurons == rows(nn{layer}.forward_weights), ['in layer: ' int2str(layer)]);
        
        assert(nn{layer}.number_of_neurons_in_previous_layer + 1 == rows(nn{layer}.backward_weights), ['in layer: ' int2str(layer)]);
        
        assert(nn{layer}.number_of_neurons == columns(nn{layer}.backward_weights), ['in layer: ' int2str(layer)]);

    end
end