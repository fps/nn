function ret = optimize(X, Y, w, rate)
    ret = w;
    prediction = model(w, X);
    error = Y - prediction;
    
    % sum along rows (DIM=2)
    ret = ret + rate * sum(error .* X, 2);
end