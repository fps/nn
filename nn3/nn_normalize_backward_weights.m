 
function nn_new = nn_normalize_backward_weights(nn, lambda)
    nn_new = nn;
    for layer = 1:rows(nn_new)
        bw = nn_new{layer}.backward_weights;
        nn_new{layer}.backward_weights /= sqrt(max(abs(eig(bw'*bw))));
    end
end