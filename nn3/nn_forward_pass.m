function nn_out = nn_forward_pass(nn, x)
    nn_out = nn;
    a = x;
    for layer = 1:rows(nn)
        layer_input = zeros(columns(nn{layer}{1,3}));
        if layer == 1
            layer_input = x
        else
            layer_input = nn_out{layer-1}{1,4}
        end
        nn_out{layer}{1,4} = nn{layer}{1,1}(nn{layer}{1,3} * layer_input);
    end
end
