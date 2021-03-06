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
for file in readdir(srcdir()); include(joinpath(srcdir(), file)); end

begin
    ciphers, ciphersdir = produce_or_load(datadir(), NamedTuple(), prefix="ciphers33", suffix="jld2") do config
        download("https://nextcloud.sdu.dk/index.php/s/Zzjcqjopy5cTawn/download/data_33.Rdata", datadir("ciphers33.RData"))
        ciphers = load(datadir("ciphers33.RData"))["ciphers"]
        @strdict ciphers
    end
    ciphers = ciphers["ciphers"]
end


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


#?? Decision trees
using DecisionTree, MLJ
Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
PCA = MLJ.@load PCA pkg=MultivariateStats

# models("DecisionTree")
# MLJ.doc("DecisionTreeClassifier", pkg="DecisionTree")
# info(Tree)

# MLJ.doc("PCA", pkg="MultivariateStats")

params = (step = 10, parts_train=1, parts_test=1)
tts = TrainTestSplit(pictures[begin:params.step:end], params.parts_train//params.parts_train)
# traindata = tts.train|>datamat|>transpose|>MLJ.table
# testdata = tts.test|>datamat|>transpose|>MLJ.table
# trainlabels = coerce(tts|>trainclasses, Multiclass)
# testlabels = coerce(tts|>testclasses, Multiclass)

#?? Test for train and test
#?? Vary tree depth (flexibility) instead of nPCs

#=
input_scitype =
     Table{<:Union{AbstractVector{<:Count}, AbstractVector{<:OrderedFactor}, AbstractVector{<:Continuous}}},
 target_scitype = AbstractVector{<:Finite},
=#
# X, y = @load_iris  # To check format
##
using Statistics: mean
using MultivariateStats
# info(PCA)

begin #?? Seeing how many PCs is good:
    results = DataFrame(n_PCs=Int[], acc=Float64[], prec=Float64[], rec=Float64[])
    for outdim in [1, (5:5:50)...]
        @show outdim
        mach_pca = machine(PCA(maxoutdim=outdim), traindata)
        MLJ.fit!(mach_pca)
        traindata_projected = MLJ.transform(mach_pca, traindata)
        testdata_projected = MLJ.transform(mach_pca, testdata)

        mach_tree = machine(Tree(), traindata_projected, trainlabels)
        MLJ.fit!(mach_tree)
        y?? = MLJ.predict_mode(mach_tree, testdata_projected)

        local_result = [metric()(y??, testlabels) for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
        pushfirst!(local_result, outdim)
        push!(results, local_result)
        # for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]
            # print(lpad(string(metric, " :  "), 30))
            # round(metric()(y??, testlabels), sigdigits=5) |> println
        #     push!(results, round(metric()(y??, testlabels), sigdigits=5) |> println
        # end
    end
end
results
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Number of PCs", ylabel="Metric", title="Decision tree classifier\n$(length(pictures)??params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split")
    scatterlines!(ax, results[:, 1], results[:, 2], label="Accuracy")
    scatterlines!(ax, results[:, 1], results[:, 3], label="Precision")
    scatterlines!(ax, results[:, 1], results[:, 4], label="Recall")
    axislegend(position=(1, 0))
    fig
end

##

begin #?? Zooming in on optimal number of PCs. 16-20??? is best
    results2 = DataFrame(n_PCs=Int[], acc=Float64[], prec=Float64[], rec=Float64[])
    for outdim in 12:20
        @show outdim
        mach_pca = machine(PCA(maxoutdim=outdim), traindata)
        MLJ.fit!(mach_pca)
        traindata_projected = MLJ.transform(mach_pca, traindata)
        testdata_projected = MLJ.transform(mach_pca, testdata)

        mach_tree = machine(Tree(), traindata_projected, trainlabels)
        MLJ.fit!(mach_tree)
        y?? = MLJ.predict_mode(mach_tree, testdata_projected)

        local_result = [metric()(y??, testlabels) for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
        pushfirst!(local_result, outdim)
        push!(results2, local_result)
    end
end
results2
begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Number of PCs", ylabel="Metric", title="Decision tree classifier\n$(length(pictures)??params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split")
    scatterlines!(ax, results2[:, 1], results2[:, 2], label="Accuracy")
    scatterlines!(ax, results2[:, 1], results2[:, 3], label="Precision")
    scatterlines!(ax, results2[:, 1], results2[:, 4], label="Recall")
    axislegend(position=(1, 0))
    fig
end

##
info(Tree)
begin  #?? Varying tree depth
    results3 = DataFrame(treedepth=Int[], acc=Float64[], testset=Bool[])
    for depth in 1:20
        @show depth
        mach_pca = machine(PCA(maxoutdim=16), traindata)
        MLJ.fit!(mach_pca)
        traindata_projected = MLJ.transform(mach_pca, traindata)
        testdata_projected = MLJ.transform(mach_pca, testdata)

        mach_tree = machine(Tree(max_depth=depth), traindata_projected, trainlabels)
        MLJ.fit!(mach_tree)
        y??_test  = MLJ.predict_mode(mach_tree,  testdata_projected)
        y??_train = MLJ.predict_mode(mach_tree, traindata_projected)

        acc_test =  Accuracy()(y??_test , testlabels)
        acc_train = Accuracy()(y??_train, trainlabels)
        push!(results3, [depth, acc_test, true])
        push!(results3, [depth, acc_train, false])
    end
end

begin
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Tree depth", ylabel="Accuracy", title="Decision tree classifier\n$(length(pictures)??params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split")
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
GLMakie.inline!(true)
# for treedepth ??? 2:2:10, n_PCs ??? [5, 10, 15, 20]
using ProgressMeter


for treedepth ??? 1:2:20
# let treedepth = 3
    n_PCs = 16
    n_batches = 10
    my_pics = pictures[begin:1:end]
    batchinds = batch(shuffle(eachindex(my_pics)), n_batches, false)
    alldata = hcat(my_pics.data...)' |> MLJ.table
    alllabels = coerce(my_pics.class, Multiclass)
    global result_each_batch = Vector{Vector}(undef, n_batches)

    @showprogress for i in 1:n_batches
        
        testdata   = selectrows(alldata,   batchinds[i])
        testlabels = selectrows(alllabels, batchinds[i])

        other_inds = vcat(deleteat!(copy(batchinds), i)...)
        traindata   = selectrows(alldata,   other_inds)
        trainlabels = selectrows(alllabels, other_inds)

        mach_pca = machine(PCA(maxoutdim=n_PCs), traindata)
        MLJ.fit!(mach_pca, verbosity=0)
        traindata_projected = MLJ.transform(mach_pca, traindata)
        testdata_projected = MLJ.transform(mach_pca, testdata)

        mach_tree_PCA = machine(Tree(max_depth=treedepth), traindata_projected, trainlabels)
        MLJ.fit!(mach_tree_PCA, verbosity=0)

        mach_tree = machine(Tree(max_depth=treedepth), traindata, trainlabels)
        MLJ.fit!(mach_tree, verbosity=0)
        
        accs = Vector{Float64}(undef, 4)
        accs[1] = mean(predict_mode(mach_tree,      testdata) .== testlabels)
        accs[2] = mean(predict_mode(mach_tree,     traindata) .== trainlabels)
        accs[3] = mean(predict_mode(mach_tree_PCA,  testdata_projected) .== testlabels)
        accs[4] = mean(predict_mode(mach_tree_PCA, traindata_projected) .== trainlabels)

        result_each_batch[i] = accs
    end
    result_each_batch = result_each_batch .|> identity

    result_each_batch_cat = hcat(result_each_batch...)
    accs1 = result_each_batch_cat[1, :]
    accs2 = result_each_batch_cat[2, :]
    accs3 = result_each_batch_cat[3, :]
    accs4 = result_each_batch_cat[4, :]
    fig = Figure()
    ax = Axis(fig[1, 1])
    whiskerwidth=10
    spacing = 0.03
    errorbars!(ax, [1-spacing], [mean(accs1)], [std(accs1)]; whiskerwidth, label="No PCA")
    errorbars!(ax, [2-spacing], [mean(accs2)], [std(accs2)]; whiskerwidth, label="No PCA")
    errorbars!(ax, [1+spacing], [mean(accs3)], [std(accs3)]; whiskerwidth, label="PCA")
    errorbars!(ax, [2+spacing], [mean(accs4)], [std(accs4)]; whiskerwidth, label="PCA")
    scatter!(ax,   [1-spacing], [mean(accs1)]; color=Cycled(1), label="No PCA")
    scatter!(ax,   [2-spacing], [mean(accs2)]; color=Cycled(1), label="No PCA")
    scatter!(ax,   [1+spacing], [mean(accs3)]; color=Cycled(2), label="PCA")
    scatter!(ax,   [2+spacing], [mean(accs4)]; color=Cycled(2), label="PCA")
    axislegend(position=(1, 0), merge=true)
    xlims!(0.5, 2.5)
    ax.ylabel = "Accuracy"
    ax.title = "Number of PCs: $n_PCs\nTree depth: $treedepth"
    ax.xticks = ([1, 2], ["Test", "Train"])
    display(fig)
end

##

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
    ??s = mean.(eval_res.per_fold)
    ??s = std.(eval_res.per_fold)
    ax, plt1 = errorbars(fig[1, 1], eachindex(eval_res.per_fold), ??s, ??s, whiskerwidth=15, color=:green, label="With PCA")
    plt2 = scatter!(ax, eachindex(eval_res.per_fold), ??s, color=:green, label="With PCA")
    ??s_noPCA = mean.(eval_res_noPCA.per_fold)
    ??s_noPCA = std.(eval_res_noPCA.per_fold)
    plt3 = errorbars!(ax, eachindex(eval_res_noPCA.per_fold), ??s_noPCA, ??s_noPCA, whiskerwidth=15, color=:red, label="Without PCA")
    plt2 = scatter!(ax, eachindex(eval_res_noPCA.per_fold), ??s_noPCA, color=:red, label="Without PCA")
    ax.xticks = (1:3, ["Accuracy", "Precision", "Recall"])
    xlims!(0.8, 3.2)
    axislegend(merge=true, position=(1, 0.55))
    fig
end
eval_res.train_test_rows
eval_res.report_per_fold
#??  Compute the optimal decision point for the first 5 PCAs of a dataset (e.g. a single person) and 
#??  compute the information gain associated to it (plot 5 graphs, one for each component, and show 
#??  the highest information gain). See slides for how to compute information gain.

#??  Compute a decision tree for the digit classification and visualize it. 
#??  You can use ???rpart??? for creating a tree and ???rpart.plot??? for visualizing the tree.

#??  Using the full data set (i.e. dataset from multiple people), evaluate a trained 
#??  decision tree using cross validation. Try to train a tree with PCA, and without PCA 
#??  (raw data). Discuss the important parameters.

#?? Random forests
#?? Create a Random Forest classifier and evaluate it using cross validation. Discuss the critical parameters of ???randomForest??? (e.g. number and depth of trees)
#?? Random forest with first 5 principal components