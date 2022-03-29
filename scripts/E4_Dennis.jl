using Pkg
paths = [raw"C:\Users\Asbjo\OneDrive - Syddansk Universitet\Machine\GitHubulub\StatisticalMachineLearning_drwatson", raw"C:\Users\Dennis Bal\GeekyStuff\Julia\6. Sem\Statistical Machine Learning\drwatson", "/home/legolas/StatisticalMachineLearning_drwatson"]
function activate_dir()
    for path in paths
        if path|>isdir
            path|>Pkg.activate
            Pkg.instantiate()
            return nothing
        end
    end
    "Project not activated. Ensure that the repository is cloned locally to one of the locations in `paths`." |> error
end
activate_dir()

using DrWatson
for file in readdir(srcdir())
    include(joinpath(srcdir(), file))
end

begin
    ciphers, ciphersdir = produce_or_load(datadir(), NamedTuple(), prefix="ciphers33", suffix="jld2") do config
        download("https://nextcloud.sdu.dk/index.php/s/Zzjcqjopy5cTawn/download/data_33.Rdata", datadir("ciphers33.RData"))
        ciphers = load(datadir("ciphers33.RData"))["ciphers"]
        @strdict ciphers
    end
    ciphers = ciphers["ciphers"]
end

using BenchmarkHistograms
pictures = Picture.(ciphers|>eachrow) |> remove_constant |> x->sort(x, by=y->y.class)
person(ID) = filter(x -> x.ID == ID, pictures)
numbersearch(pics::Vector{<:Picture}, nr) = (filter(pic -> pic.class == nr, pics))
nothing

##!======================================================!##
#? Discretize the range of values of principle componten coefficient in e.g. 200


#¤ Decision trees
using DecisionTree, MLJ

MLJ.doc("DecisionTreeClassifier", pkg="DecisionTree")
Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
info(Tree)

MLJ.doc("PCA", pkg="MultivariateStats")
PCA = MLJ.@load PCA pkg=MultivariateStats

params = (step = 1, parts_train=1, parts_test=1)
tts = TrainTestSplit(pictures[begin:params.step:end], params.parts_train//params.parts_train)
traindata = tts.train|>datamat|>transpose|>MLJ.table
testdata = tts.test|>datamat|>transpose|>MLJ.table
trainlabels = coerce(tts|>trainclasses, Multiclass)
testlabels = coerce(tts|>testclasses, Multiclass)


#=
input_scitype =
     Table{<:Union{AbstractVector{<:Count}, AbstractVector{<:OrderedFactor}, AbstractVector{<:Continuous}}},
 target_scitype = AbstractVector{<:Finite},
=#
# X, y = @load_iris  # To check format
##
using Statistics: mean

info(PCA)
begin
    results = DataFrame(n_PCs=Int[], acc=Float64[], prec=Float64[], rec=Float64[])
    for outdim in [1, (5:5:50)...]
        @show outdim
        mach_pca = machine(PCA(maxoutdim=outdim), traindata)
        MLJ.fit!(mach_pca)
        mach_pca
        traindata_projected = MLJ.transform(mach_pca, traindata)
        testdata_projected = MLJ.transform(mach_pca, testdata)

        mach_tree = machine(Tree(), traindata_projected, trainlabels)
        MLJ.fit!(mach_tree)
        ŷ = MLJ.predict_mode(mach_tree, testdata_projected)

        local_result = [metric()(ŷ, testlabels) for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
        pushfirst!(local_result, outdim)
        push!(results, local_result)
        # for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]
            # print(lpad(string(metric, " :  "), 30))
            # round(metric()(ŷ, testlabels), sigdigits=5) |> println
        #     push!(results, round(metric()(ŷ, testlabels), sigdigits=5) |> println
        # end
    end
end
results

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Number of PCs", ylabel="Metric", title="Decision tree classifier\n$(length(pictures)÷params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split")
    scatterlines!(ax, results[:, 1], results[:, 2], label="Accuracy")
    scatterlines!(ax, results[:, 1], results[:, 3], label="Precision")
    scatterlines!(ax, results[:, 1], results[:, 4], label="Recall")
    axislegend(position=(1, 0))
    fig
end
# report(mach)
print_tree(mach.fitresult[1])
Accuracy()
MLJ.evaluate(mach, testdata, testlabels)
MLJ.evaluate!(mach, measures=[accuracy])
print_tree

##

#¤  Compute the optimal decision point for the first 5 PCAs of a dataset (e.g. a single person) and 
#¤  compute the information gain associated to it (plot 5 graphs, one for each component, and show 
#¤  the highest information gain). See slides for how to compute information gain.

#¤  Compute a decision tree for the digit classification and visualize it. 
#¤  You can use “rpart” for creating a tree and “rpart.plot” for visualizing the tree.

#¤  Using the full data set (i.e. dataset from multiple people), evaluate a trained 
#¤  decision tree using cross validation. Try to train a tree with PCA, and without PCA 
#¤  (raw data). Discuss the important parameters.
