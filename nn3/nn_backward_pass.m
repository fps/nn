 
function nn_out = nn_backward_pass(nn, targets, rate)
    nn_out = nn;
    number_of_layers = rows(nn);
    number_of_targets = columns(targets);
    
    for backwards_layer = 0:(number_of_layers - 1)
        layer = number_of_layers - backwards_layer;
        
        if layer != number_of_layers
            % break;
        end
        
        layer = layer
        
        if layer == number_of_layers
            'errors in last layer'
            nn_out{layer}.errors = (targets - nn_out{layer}.activations) .* nn_out{layer}.df(nn_out{layer}.sums);
        end
        
        nn_assert_consistency(nn);

        if layer > 1
            nn_out{layer-1}.errors = zeros(nn_out{layer-1}.number_of_neurons, columns(targets));
            
            % size(nn_out{layer-1}.errors)
            
            for target = 1:columns(targets)
                previous_activations_derivative = nn_out{layer-1}.df(nn_out{layer-1}.sums(:,target));
                
                current_backward_weights = nn_out{layer}.backward_weights;
                
                current_errors = nn_out{layer}.errors(:, target);
                
                % size(current_errors)
                
                nn_out{layer-1}.errors(:, target) = (current_backward_weights * current_errors) .* previous_activations_derivative;
            end
        end

        nn_assert_consistency(nn);

        % update weights
        for target = 1:columns(targets)
            % size(nn_out{layer-1}.activations(:, target))
            
            delta_w = (rate / number_of_targets) * (nn_out{layer}.errors(:,target) .* nn_out{layer}.inputs(:,target)');

            nn_out{layer}.forward_weights += delta_w;
        end
        
        nn_assert_consistency(nn);
    end
end