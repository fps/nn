function test()
    more off
    [X, Y] = generate()

    w = zeros(2, 3)

    for i = 1:50
        w = optimize(X, Y, w, 0.01)
        plot3(X(1,:), X(2,:), (w*X)(1,:), ".", X(1,:), X(2,:), Y(1,:), "*")
        sleep(0.1)
    end
end
