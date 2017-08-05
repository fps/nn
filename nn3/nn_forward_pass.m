function nn_out = nn_forward_pass(nn, x)
    nn_out = nn;
    for layer = 1:rows(nn)
        layer_input = [];
        if layer == 1
            layer_input = x;
        else
            layer_input = nn_out{layer-1}.activations;
        end
        nn_out{layer}.activations = nn{layer}.f(nn{layer}.forward_weights * layer_input);
    end
end
