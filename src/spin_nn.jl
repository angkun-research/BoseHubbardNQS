using PythonCall

const _pynet = Ref{Py}()

function pynet()
    if !isassigned(_pynet)
        sys = pyimport("sys")
        #sys.path.insert(0, @__DIR__)
        python_src = joinpath(@__DIR__, "python")
        sys.path.insert(0, python_src)
        _pynet[] = pyimport("simple_spin_nn")
    end
    return _pynet[]
end

function load_nn(path::String="data/spin_net.pt")
    return pynet().load_model(path)
end

function nn_value(model, spin_config::AbstractVector{<:Integer})
    cfg = collect(Int.(spin_config))
    y = pynet().predict_scalar(model, cfg)
    return pyconvert(Float64, y)
end

function ansatz_amplitude(model, sigma::Vector{Int})
    return nn_value(model, sigma)
end