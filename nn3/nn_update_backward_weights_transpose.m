 
function nn_out = nn_update_backward_weights_transpose(nn)
    nn_out = nn;
    for layer = 1:rows(nn)
        nn_out{layer}.backward_weights = nn{layer}.forward_weights';
    end
end