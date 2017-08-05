 
function new_nn = nn_initialize_weights_gaussian(nn)
    new_nn = nn;
    
    for layer = 1:rows(new_nn)
        new_nn{layer} = setfield(new_nn{layer}, 'forward_weights', randn(size(getfield(new_nn{layer}, 'forward_weights'))));
    end
end