module BoseHubbardNQS

using PythonCall

#include("spin_nn.jl")
#export load_nn, nn_value, ansatz_amplitude

include("Bose_Hubbard_NQS.jl")

export load_nn, nn_value, ansatz_log_amplitude

end