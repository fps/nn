% adds a layer created e.g. by nn_create_layer
% to the nn
function nn_out = nn_add_layer(nn, layer)
    nn_out = nn;
    number_of_layers = rows(nn);
    
    if number_of_layers > 0
        assert(columns(getfield(layer, 'forward_weights')) == rows(getfield(nn_out{number_of_layers}, 'forward_weights')) + 1, "mismatching weight matrix dimensions")
    end
    
    nn_out{number_of_layers + 1,1} = layer;
    
    nn_assert_consistency(nn_out);
end