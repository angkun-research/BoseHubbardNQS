using Test
using BoseHubbardNQS
using PythonCall

const bh = BoseHubbardNQS.bhnet()
const torch = pyimport("torch")
const np = pyimport("numpy")
const models = pyimport("pythonTest.bosehubbard2d.models")
torch.set_num_threads(1)

@testset "1D legacy scalar and batch transport" begin
    mktempdir() do directory
        torch.manual_seed(123)
        original = bh.build_model("FCNet", 3, 2, 4)
        path = joinpath(directory, "one.pt")
        torch.save(pydict("arch"=>"FCNet", "n_sites"=>3, "nmax"=>2,
                        "hidden_dim"=>4, "state_dict"=>original.state_dict()), path)
        m = load_nn(path)
        @test m isa Py
        configs = [0 2 0; 1 0 0; 2 1 0]
        gold = [0.3105006217956543, 0.308648943901062, 0.35193121433258057]
        # Allow float32 rounding differences between CPU kernel implementations.
        @test [nn_value(m, c) for c in eachcol(configs)] ≈ gold rtol=0 atol=1e-7
        @test ansatz_log_amplitude(m, configs[:,1]) ≈ gold[1] rtol=0 atol=1e-7
        @test nn_values(m, configs; chunk_size=2) ≈ gold rtol=2e-6 atol=2e-6
        @test nn_values(m, @view(configs[:,3:-1:1])) ≈ reverse(gold) rtol=2e-6 atol=2e-6
        @test nn_values(m, zeros(Int, 3, 0)) == Float64[]
        @test_throws PyException nn_value(m, [0,1])
        @test_throws PyException nn_values(m, fill(-1, 3, 2))
        @test_throws PyException nn_values(m, configs; chunk_size=0)
        @test_throws PyException nn_values(m, configs; nmax=3)
        @test_throws MethodError nn_values(m, zeros(3,2))
        @test pyconvert(String, load_nn(path; device="cpu").net[0].weight.device.type) == "cpu"
        @test_throws PyException load_nn(path; device="cuda:99999")
        @test_throws PyException load_nn(path; device="cuda:256")
    end
end

@testset "2D generated checkpoints: all architectures, dtype and geometry" begin
    source = pyconvert(String, pyimport("builtins").str(models.ROOT))
    mktempdir(source; prefix=".julia-bridge-") do directory
        configs = reshape(mod.(0:23,3),6,4)
        arches = pyconvert(Vector{String}, pyimport("builtins").list(models.ARCH_BUILDERS))
        for arch in arches, dtype in ("float32", "float64")
            boundaries = startswith(arch,"Group") ? [(true,true)] : [(false,false),(true,true),(false,true)]
            for (px,py) in boundaries
                settings = arch == "JastrowNet" ? (;) : arch == "RBMNet" ? (;n_hidden=4) : (;hidden_dim=4)
                original = models.build_model(arch,3,2,2; pbc_x=px,pbc_y=py,settings...).to(dtype=getproperty(torch,Symbol(dtype))).eval()
                path = joinpath(directory,"$(arch)-$(dtype)-$(px)-$(py).pt")
                models.save_model(path,original;Ntot=4,t=.7,U=2.3,mu=.2)
                m = load_nn_2d(path)
                @test m isa NQS2D
                @test m.metadata["Lx"] == 3 && m.metadata["Ly"] == 2
                @test m.metadata["pbc_x"] == px && m.metadata["pbc_y"] == py
                @test m.metadata["dtype"] == dtype
                @test m.metadata["Ntot"] == 4 && m.metadata["U"] == 2.3
                @test m.metadata["architecture_settings"] isa Dict
                @test m.metadata["spatial_symmetries"] isa Vector
                expected = pyconvert(Vector{Float64}, BoseHubbardNQS.nqs2d().predict_values(original,Py(configs)))
                @test nn_values(m,configs;chunk_size=3) ≈ expected rtol=2e-6 atol=2e-6
                @test [nn_value(m,c) for c in eachcol(configs)] ≈ expected rtol=2e-6 atol=2e-6
                @test ansatz_log_amplitude(m,configs[:,1]) == nn_value(m,configs[:,1])
                @test nn_values(m,@view(configs[:,4:-1:1])) ≈ reverse(expected) rtol=2e-6 atol=2e-6
                @test isempty(nn_values(m,zeros(Int,6,0)))
                @test_throws PyException nn_values(m,zeros(Int,5,1))
                @test_throws PyException nn_value(m,fill(3,6))
                @test_throws PyException load_nn(path)
            end
        end
    end
end

@testset "isolated 2D workflows through Julia keywords" begin
    source = pyconvert(String, pyimport("builtins").str(models.ROOT))
    mktempdir(source; prefix=".julia-workflow-") do directory
        torch.manual_seed(45)
        state = torch.random.get_rng_state().clone()
        edpath = joinpath(directory,"ed.pt")
        ed = train_ed_2d(; Lx=3,Ly=1,Ntot=2,Nmax=2,pbc=false,epochs=1,width=4,
                          eval_every=1,device="cpu",output=relpath(edpath))
        @test ed["report"]["status"] == "completed"
        @test ed["paths"]["checkpoint"] == edpath
        @test ed["report"] isa Dict
        @test ed["report"]["history"] isa Vector
        @test ed["report"]["history"][1] isa Dict
        @test ed["report"]["command"][1] == pyconvert(String,pyimport("sys").executable)
        @test isfile(ed["paths"]["checkpoint"])
        @test pyconvert(Bool,torch.equal(state,torch.random.get_rng_state()))
        @test load_nn_2d(edpath).metadata["pbc_x"] == false
        state = torch.random.get_rng_state().clone()
        vmcpath = joinpath(directory,"vmc.pt")
        vmc = train_vmc_2d(; Lx=3,Ly=1,Ntot=2,Nmax=2,pbc=false,epochs=1,hidden_dim=4,
                            walkers=2,steps=1,burnin=0,device="cpu",output=vmcpath)
        @test isfile(vmc["paths"]["checkpoint"])
        measured = evaluate_2d(; checkpoint=vmcpath,walkers=2,steps=2,burnin=0,
                                block_length=1,device="cpu",pair=[[0,1],[1,0]],
                                output=joinpath(directory,"eval.json"))
        @test measured["report"]["status"] == "completed"
        @test isfile(measured["paths"]["raw_samples"])
        @test pyconvert(Bool,torch.equal(state,torch.random.get_rng_state()))
        @test_throws PyException train_ed_2d(; output=edpath,epochs=1)
        @test_throws PyException train_vmc_2d(; output="/tmp/bhnqs-outside.pt",epochs=1)
    end
end

if get(ENV,"BHNQS_TEST_CUDA","0") == "1"
    @testset "actual CUDA bridge" begin
        @test pyconvert(Bool,torch.cuda.is_available())
        source = pyconvert(String, pyimport("builtins").str(models.ROOT))
        mktempdir(source; prefix=".julia-cuda-") do directory
            for arch in pyconvert(Vector{String},pybuiltins.list(models.ARCH_BUILDERS)), dtype in ("float32","float64")
                settings = arch == "JastrowNet" ? (;) : arch == "RBMNet" ? (;n_hidden=4) : (;hidden_dim=4)
                original = models.build_model(arch,3,2,2;settings...).to(dtype=getproperty(torch,Symbol(dtype)))
                path = joinpath(directory,"cuda-$(arch)-$(dtype).pt")
                models.save_model(path,original)
                cpu = load_nn_2d(path)
                gpu = load_nn_2d(path;device="cuda:0")
                @test pyconvert(String,pyimport("builtins").next(gpu.model.parameters()).device.type) == "cuda"
                cfg = reshape(mod.(0:23,3),6,4)
                @test nn_values(gpu,cfg) ≈ nn_values(cpu,cfg) rtol=3e-5 atol=3e-5
                @test nn_value(gpu,cfg[:,1]) ≈ nn_value(cpu,cfg[:,1]) rtol=3e-5 atol=3e-5
            end
            for arch in pyconvert(Vector{String},pybuiltins.list(bh.ARCH_BUILDERS))
                original = bh.build_model(arch,6,2,4)
                path = joinpath(directory,"1d-$(arch).pt")
                torch.save(pydict("arch"=>arch,"n_sites"=>6,"nmax"=>2,"hidden_dim"=>4,
                                  "state_dict"=>original.state_dict()),path)
                cpu = load_nn(path)
                gpu = load_nn(path;device="cuda:0")
                cfg = reshape(mod.(0:23,3),6,4)
                @test pyconvert(String,pybuiltins.next(gpu.parameters()).device.type) == "cuda"
                @test nn_values(gpu,cfg) ≈ nn_values(cpu,cfg) rtol=3e-5 atol=3e-5
                @test nn_value(gpu,cfg[:,1]) ≈ nn_value(cpu,cfg[:,1]) rtol=3e-5 atol=3e-5
            end
        end
    end
end
