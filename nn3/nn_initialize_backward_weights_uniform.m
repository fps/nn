 
 
function new_nn = nn_initialize_backward_weights_uniform(nn, range)
    new_nn = nn;
    
    for layer = 1:rows(new_nn)
        new_nn{layer}.backward_weights = range * (2 * rand(size(new_nn{layer}.forward_weights')) - 1);
    end
end