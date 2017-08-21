 
function nn_new = nn_reinitialize_saturated_relus(nn, var)
    nn_new = nn;
    reinit = 0;
    for layer = 1:rows(nn)
        for neuron = 1:nn{layer}.number_of_neurons
            if sum(nn{layer}.activations(neuron, :) != 0) == 0
                nn_new{layer}.forward_weights(neuron, :) = sqrt(var) * randn(1, nn{layer}.number_of_neurons_in_previous_layer + 1);
                reinit += 1;
            end
        end
    end
    reinit
end