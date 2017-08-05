 
function nn_assert_consistency(nn)
    for layer = 1:rows(nn)
        assert(nn{layer}.number_of_neurons == rows(nn{layer}.forward_weights), 'weight dimension # of neurons mismatch');
    end
end