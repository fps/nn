 
function nn_new = nn_sgd(nn, rate)
    nn_new = nn;
    for layer = 1:rows(nn)
        nn_new{layer}.forward_weights += -rate * nn_new{layer}.gradients;
    end
end