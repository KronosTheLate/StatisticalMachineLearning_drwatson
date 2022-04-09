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
numbersearch(pics::Vector{<:Picture}, nr) = filter(pic -> pic.class == nr, pics)

using Distributions
function myqqplot(observations, dist = Normal; title = "QQ plot")
    n = length(observations)
    xlabel = (dist == Normal ? "Fitted normal distribution values" : "Theoretical distribution")
    xvals = [quantile(fit(dist, observations), i / n) for i = 1:n-1]
    yvals = [quantile(observations, i / n) for i = 1:n-1]
    scatter(xvals, yvals, label = "Quantile ",
        axis = (xlabel = xlabel, ylabel = "Observation values", title = title)
    )
    coeffs = [ones(length(xvals)) xvals] \ yvals  # The same as a least squares fit.
    lines!(xvals, x -> coeffs[1] + coeffs[2] * x, label = "Least squares linear fit")
    axislegend(position = (1, 0))
    current_figure()
end

nothing


##!======================================================!##
#? Discretize the range of values of principle componten coefficient in e.g. 200


#¤ Decision trees
using DecisionTree, MLJ
models("DecisionTree")
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

#¤ Test for train and test
#¤ Vary tree depth (flexibility) instead of nPCs

#=
input_scitype =
     Table{<:Union{AbstractVector{<:Count}, AbstractVector{<:OrderedFactor}, AbstractVector{<:Continuous}}},
 target_scitype = AbstractVector{<:Finite},
=#
# X, y = @load_iris  # To check format
##
using Statistics: mean

info(PCA)

begin #¤ Seeing how many PCs is good:
    results = DataFrame(n_PCs=Int[], acc=Float64[], prec=Float64[], rec=Float64[])
    for outdim in [1, (5:5:50)...]
        @show outdim
        mach_pca = machine(PCA(maxoutdim=outdim), traindata)
        MLJ.fit!(mach_pca)
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

##

begin #¤ Zooming in on optimal number of PCs. 16-20… is best
    results2 = DataFrame(n_PCs=Int[], acc=Float64[], prec=Float64[], rec=Float64[])
    for outdim in 12:20
        @show outdim
        mach_pca = machine(PCA(maxoutdim=outdim), traindata)
        MLJ.fit!(mach_pca)
        traindata_projected = MLJ.transform(mach_pca, traindata)
        testdata_projected = MLJ.transform(mach_pca, testdata)

        mach_tree = machine(Tree(), traindata_projected, trainlabels)
        MLJ.fit!(mach_tree)
        ŷ = MLJ.predict_mode(mach_tree, testdata_projected)

        local_result = [metric()(ŷ, testlabels) for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
        pushfirst!(local_result, outdim)
        push!(results2, local_result)
    end
end
results2
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Number of PCs", ylabel="Metric", title="Decision tree classifier\n$(length(pictures)÷params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split")
    scatterlines!(ax, results2[:, 1], results2[:, 2], label="Accuracy")
    scatterlines!(ax, results2[:, 1], results2[:, 3], label="Precision")
    scatterlines!(ax, results2[:, 1], results2[:, 4], label="Recall")
    axislegend(position=(1, 0))
    fig
end

##
info(Tree)
begin  #¤ Varying tree depth
    results3 = DataFrame(treedepth=Int[], acc=Float64[], testset=Bool[])
    for depth in 1:20
        @show depth
        mach_pca = machine(PCA(maxoutdim=16), traindata)
        MLJ.fit!(mach_pca)
        traindata_projected = MLJ.transform(mach_pca, traindata)
        testdata_projected = MLJ.transform(mach_pca, testdata)

        mach_tree = machine(Tree(max_depth=depth), traindata_projected, trainlabels)
        MLJ.fit!(mach_tree)
        ŷ_test  = MLJ.predict_mode(mach_tree,  testdata_projected)
        ŷ_train = MLJ.predict_mode(mach_tree, traindata_projected)

        acc_test =  Accuracy()(ŷ_test , testlabels)
        acc_train = Accuracy()(ŷ_train, trainlabels)
        push!(results3, [depth, acc_test, true])
        push!(results3, [depth, acc_train, false])
    end
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Tree depth", ylabel="Accuracy", title="Decision tree classifier\n$(length(pictures)÷params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split")
    let
        data = filter(row->row.testset, results3)
        scatterlines!(ax, data.treedepth, data.acc, label="Test")
    end
    let
        data = filter(row->!row.testset, results3)
        scatterlines!(ax, data.treedepth, data.acc, label="Train")
    end
    axislegend("Validation with", position=(1, 0.2))
    fig
end
results3.testset

##
# report(mach_tree)
mach_pca = machine(PCA(maxoutdim=16), traindata)
MLJ.fit!(mach_pca)
mach_pca
traindata_projected = MLJ.transform(mach_pca, traindata)
testdata_projected = MLJ.transform(mach_pca, testdata)

# info(Tree)
mach_tree = machine(Tree(max_depth=16), traindata_projected, trainlabels)
MLJ.fit!(mach_tree)
ŷ = MLJ.predict_mode(mach_tree, testdata_projected)
mean(ŷ .== testlabels)
# print_tree(mach_tree.fitresult[1])
# typeof(mach_tree.fitresult[1])
##
begin
    n_batches = 10
    metric = Accuracy()
    testbachinds = batch(shuffle(eachindex(testlabels)), n_batches, false)
    testbatches = selectrows.([testdata_projected], testbachinds)
    accs = Vector{Float64}(undef, n_batches)
    for i in eachindex(testbachinds)
        testdata_batch = selectrows(testdata_projected, testbachinds[i])
        testlabel_batch = selectrows(testlabels, testbachinds[i])
        ŷ = MLJ.predict_mode(mach_tree, testdata_batch)
        y = testlabel_batch
        accs[i] = metric(ŷ, y)
    end
    accs
end





eval_res = evaluate!(mach_tree, testdata, testlabels,
         resampling=CV(nfolds=10),
         measure=[Accuracy(), MulticlassPrecision(), MulticlassTruePositiveRate()]
)

mach_tree_noPCA = machine(Tree(max_depth=16), traindata, trainlabels)
MLJ.fit!(mach_tree_noPCA)

eval_res_noPCA = evaluate!(mach_tree_noPCA,# testdata, testlabels,
         resampling=CV(nfolds=10),
         measure=[Accuracy(), MulticlassPrecision(), MulticlassTruePositiveRate()]
)

myqqplot(eval_res.per_fold[3])
let
    fig = Figure()
    μs = mean.(eval_res.per_fold)
    σs = std.(eval_res.per_fold)
    ax, plt1 = errorbars(fig[1, 1], eachindex(eval_res.per_fold), μs, σs, whiskerwidth=15, color=:green, label="With PCA")
    plt2 = scatter!(ax, eachindex(eval_res.per_fold), μs, color=:green, label="With PCA")
    μs_noPCA = mean.(eval_res_noPCA.per_fold)
    σs_noPCA = std.(eval_res_noPCA.per_fold)
    plt3 = errorbars!(ax, eachindex(eval_res_noPCA.per_fold), μs_noPCA, σs_noPCA, whiskerwidth=15, color=:red, label="Without PCA")
    plt2 = scatter!(ax, eachindex(eval_res_noPCA.per_fold), μs_noPCA, color=:red, label="Without PCA")
    ax.xticks = (1:3, ["Accuracy", "Precision", "Recall"])
    xlims!(0.8, 3.2)
    axislegend(merge=true, position=(1, 0.55))
    fig
end
eval_res.train_test_rows
eval_res.report_per_fold
#¤  Compute the optimal decision point for the first 5 PCAs of a dataset (e.g. a single person) and 
#¤  compute the information gain associated to it (plot 5 graphs, one for each component, and show 
#¤  the highest information gain). See slides for how to compute information gain.

#¤  Compute a decision tree for the digit classification and visualize it. 
#¤  You can use “rpart” for creating a tree and “rpart.plot” for visualizing the tree.

#¤  Using the full data set (i.e. dataset from multiple people), evaluate a trained 
#¤  decision tree using cross validation. Try to train a tree with PCA, and without PCA 
#¤  (raw data). Discuss the important parameters.

#¤ Random forests
#¤ Create a Random Forest classifier and evaluate it using cross validation. Discuss the critical parameters of “randomForest” (e.g. number and depth of trees)
#¤ Random forest with first 5 principal components