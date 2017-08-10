 
 
function new_nn = nn_initialize_backward_weights_gaussian(nn)
    new_nn = nn;
    
    for layer = 1:rows(new_nn)
        new_nn{layer} = setfield(new_nn{layer}, 'backward_weights', randn(size(getfield(new_nn{layer}, 'backward_weights'))));
    end
end