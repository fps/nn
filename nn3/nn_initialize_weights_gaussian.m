 
function new_nn = nn_initialize_weights_gaussian(nn)
    new_nn = nn;
    
    for layer = 1:rows(new_nn)
        new_nn{layer}{1,3} = randn(size(new_nn{layer}{1,3}));
    end
end