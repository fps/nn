function test()
    more off
    [X, Y] = generate()

    w = 0

    for i = 1:500
        w = optimize(X, Y, w, 0.01)
        plot(X, w*X, ".", X, Y, "*")
        sleep(0.1)
    end
end
