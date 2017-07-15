function ret = optimize(X, Y, w, rate)
    ret = w;
    prediction = model(w, X);
    error = Y - prediction;
    
    summed_error = sum(error.**2, 2) / columns(X)
    
    % sum along rows (DIM=2)
    ret = ret + rate * error * X';
end