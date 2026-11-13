module BoseHubbardNQS

using PythonCall

#include("spin_nn.jl")
#export load_nn, nn_value, ansatz_amplitude

include("Bose_Hubbard_NQS.jl")

include("nqs_2d.jl")

export load_nn, nn_value, nn_values, ansatz_log_amplitude
export NQS2D, load_nn_2d, train_vmc_2d, train_ed_2d, evaluate_2d

end