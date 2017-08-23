 
function new_nn = nn_initialize_forward_weights_gaussian(nn, var)
    new_nn = nn;
    
    for layer = 1:rows(new_nn)
        normalization = (sqrt(var) / sqrt(new_nn{layer}.number_of_neurons_in_previous_layer));
        
        new_nn{layer}.forward_weights = normalization * randn(size(new_nn{layer}.forward_weights));
    end
end