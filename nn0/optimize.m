function ret = optimize(X, Y, w, rate)
    ret = w;
    prediction = model(w, X);
    error = Y - prediction;
    ret = ret + rate * sum(error .* X);
end