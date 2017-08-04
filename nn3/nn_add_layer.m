% adds a layer created e.g. by nn_create_layer
% to the nn
function nn_out = nn_add_layer(nn, layer)
    nn_out = nn;
    number_of_layers = rows(nn);
    nn_out{number_of_layers + 1,1} = layer;
end