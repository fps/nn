 
function nn_count_saturated_relus(nn)
    for layer = 1:rows(nn)
        for neuron = 1:nn{layer}.number_of_neurons
            %[sum(nn{layer}.activations(neuron,:) == 0, 2) / columns(nn{layer}.activations) sum(nn{layer}.activations(neuron,:) == 0, 2)]
        end
        sum(nn{layer}.activations == 0, 2)' / columns(nn{layer}.activations)
    end
end