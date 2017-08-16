 
 
function new_nn = nn_initialize_backward_weights_unity(nn)
    new_nn = nn;
    
    for layer = 1:rows(new_nn)
        new_nn{layer}.backward_weights =  ones(size(new_nn{layer}.forward_weights'));
    end
end