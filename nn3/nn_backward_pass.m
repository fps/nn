 
function nn_out = nn_backward_pass(nn, targets, rate)
    nn_out = nn;
    number_of_layers = rows(nn);
    
    
    for layer = 0:number_of_layers - 1
        backwards_layer = number_of_layers - layer;
        
        if backwards_layer == number_of_layers
            % error in last layer
            % 'errors'
            nn_out{backwards_layer}.errors = targets - nn_out{backwards_layer}.activations;
            
            % columns(targets)
            % update weights
            for target = 1:columns(targets)
                nn_out{backwards_layer}.forward_weights += rate * nn_out{backwards_layer}.errors(:,target) * (nn_out{backwards_layer}.df(nn_out{backwards_layer}.activations(:,target)) .* nn_out{backwards_layer-1}.activations(:,target)');
            end
        end
    end
end