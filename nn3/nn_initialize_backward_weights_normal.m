 
 
function new_nn = nn_initialize_backward_weights_normal(nn)
    new_nn = nn;
    
    for layer = 1:rows(new_nn)
        new_nn{layer} = setfield(new_nn{layer}, 'backward_weights', 2*rand(size(getfield(new_nn{layer}, 'backward_weights'))) - 1);
    end
end