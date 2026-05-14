module BoseHubbardNQS

using PythonCall

include("spin_nn.jl")

export load_nn, nn_value, ansatz_amplitude

end