 
function new_nn = nn_initialize_forward_weights_gaussian(nn, var)
    new_nn = nn;
    
    for layer = 1:rows(new_nn)
        new_nn{layer}.forward_weights = (sqrt(var) / sqrt(new_nn{layer}.number_of_neurons)) * randn(size(new_nn{layer}.forward_weights));
    end
end