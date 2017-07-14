function ret = optimize(X, Y, w, rate)
    ret = w;
    for sample = 1:columns(X)
        x_i = X(1, sample);
        y_i = Y(1, sample);
        prediction = model(w, x_i);
        error = y_i - prediction;
        ret = ret + rate * error * x_i;
    end
end