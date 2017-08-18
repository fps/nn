function nn_out = nn_forward_pass(nn, x)
    nn_out = nn;
    for layer = 1:rows(nn)
        layer_input = [];
        
        if layer == 1
            nn_out{layer}.inputs = [x; ones(1, columns(x))];
        else
            prev_act = nn_out{layer-1}.activations;
            
            nn_out{layer}.inputs = [prev_act; ones(1, columns(prev_act))];
        end

        nn_out{layer}.sums = nn{layer}.forward_weights * nn_out{layer}.inputs;
        nn_out{layer}.activations = nn{layer}.f(nn_out{layer}.sums);

        %nn_out{layer}.activations = nn{layer}.f(nn{layer}.forward_weights * layer_input);
    end
end
