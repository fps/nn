 
function new_nn = nn_update_backward_weights_gaussian(nn, var)
     new_nn = nn;
    for layer = 1:rows(nn)
        new_nn{layer}.backward_weights = nn{layer}.backward_weights + var * randn(size(nn{layer}.backward_weights));
    end
end