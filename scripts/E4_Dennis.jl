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
using Statistics: mean

using AlgebraOfGraphics
const AOG = AlgebraOfGraphics
set_aog_theme!()
update_theme!(resolution = (1920÷2, 1080÷3))

batch(1:33, 11, false)

##!======================================================!##
#? Discretize the range of values of principle componten coefficient in e.g. 200


#¤ Decision trees
using MLJ
using DecisionTree
using MultivariateStats  #! This possibly fixes the `size` method issues.
Tree = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
pca = MLJ.@load PCA pkg=MultivariateStats



# models("DecisionTree")
# MLJ.doc("DecisionTreeClassifier", pkg="DecisionTree")
# info(Tree)

# MLJ.doc("PCA", pkg="MultivariateStats")
shufflepics = pictures[shuffle(eachindex(pictures))]
picdata = shufflepics[begin:4:end]|>datamat|>transpose|>MLJ.table
picclasses = coerce(shufflepics.class[begin:4:end], Multiclass)

# params = (step = 10, parts_train=1, parts_test=1)
# tts = TrainTestSplit(pictures[begin:params.step:end], params.parts_train//params.parts_train)
# traindata = tts.train|>datamat|>transpose|>MLJ.table
# testdata = tts.test|>datamat|>transpose|>MLJ.table
# trainlabels = coerce(tts|>trainclasses, Multiclass)
# testlabels = coerce(tts|>testclasses, Multiclass)

#¤ Test for train and test
#¤ Vary tree depth (flexibility) instead of nPCs

#=
input_scitype =
     Table{<:Union{AbstractVector{<:Count}, AbstractVector{<:OrderedFactor}, AbstractVector{<:Continuous}}},
 target_scitype = AbstractVector{<:Finite},
=#
# X, y = @load_iris  # To check format
##

# info(PCA)


# length(picdata)
# eachindex(picdata)[begin:10:end]
mach_pca = machine(pca(pratio=0.5), picdata)
MLJ.fit!(mach_pca, verbosity=0)

# traindata_projected = MLJ.transform(mach_pca, traindata)
# testdata_projected = MLJ.transform(mach_pca, testdatsa)

# mach_tree = machine(Tree(; max_depth), traindata_projected, trainlabels)
# MLJ.fit!(mach_tree)
# ŷ = MLJ.predict_mode(mach_tree, testdata_projected)
traininds, testinds = partition(eachindex(picclasses), 0.9, rng=69)
# using StatsBase
# picclasses[traininds] |> countmap
# picclasses[testinds] |> countmap
# bla |> countmap
begin
    global results_train = DataFrame(time=Float64[], max_depth = Int[], n_PCs=Int[], acc=Float64[], prec=Float64[], rec=Float64[])
    global results_test  = DataFrame(time=Float64[], max_depth = Int[], n_PCs=Int[], acc=Float64[], prec=Float64[], rec=Float64[])
    for max_depth = 2:2:20
        
        for outdim in 5:5:40
            
            t0 = time()
            mach_pca = machine(PCA(maxoutdim=outdim), picdata)
            MLJ.fit!(mach_pca, rows=traininds)
            picdata_projected = MLJ.transform(mach_pca, picdata)

            mach_tree = machine(Tree(; max_depth), picdata_projected, picclasses)
            MLJ.fit!(mach_tree, rows=traininds)
            ŷ = MLJ.predict_mode(mach_tree, picdata_projected)
            time_elapsed = time() - t0
            # return 0
            local_result_test  = [metric()(ŷ[testinds] , picclasses[testinds])  for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
            pushfirst!(local_result_test , outdim)
            pushfirst!(local_result_test , max_depth)
            pushfirst!(local_result_test , time_elapsed)
            push!(results_test, local_result_test)


            local_result_train = [metric()(ŷ[traininds], picclasses[traininds]) for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
            pushfirst!(local_result_train, outdim)
            pushfirst!(local_result_train, max_depth)
            pushfirst!(local_result_train , time_elapsed)
            push!(results_train, local_result_train)

            # for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]
                # print(lpad(string(metric, " :  "), 30))
                # round(metric()(ŷ, testlabels), sigdigits=5) |> println
            #     push!(results, round(metric()(ŷ, testlabels), sigdigits=5) |> println
            # end
        end
    end
end
results_train
results_test

begin
draw(visual(Scatter, colormap=:thermal, colorrange=extrema(results_train.acc), markersize=40) * AOG.data(results_train) * mapping(:max_depth => "Max tree depth", :n_PCs => "Number of PCs", color=:acc=>"Accuracy"))
current_axis().title = "Reduced train data"; current_figure()
end
begin
draw(visual(Scatter, colormap=:thermal, colorrange=extrema(results_train.acc), markersize=40) * AOG.data(results_test) * mapping(:max_depth => "Max tree depth", :n_PCs => "Number of PCs", color=:acc=>"Accuracy"))
current_axis().title = "Reduced test data"; current_figure()
end
begin
    draw(visual(Scatter, colormap=:thermal, markersize=40) * AOG.data(results_train) * mapping(:max_depth => "Max tree depth", :n_PCs => "Number of PCs", color=:time=>"Time to train and classify (s)"))
    current_axis().title = "Reduced data timings"; current_figure()
end

##

let #For testing isolated case
    n_trees = 50
    bagging_fraction = 0.8

    t0 = time()
    mach_pca = machine(PCA(maxoutdim=15), picdata)
    MLJ.fit!(mach_pca, rows=traininds, verbosity=0)
    picdata_projected = MLJ.transform(mach_pca, picdata)

    ensamble = EnsembleModel(Tree(; max_depth=25); bagging_fraction, n=n_trees, acceleration=CPUThreads())
    forest_mach = machine(ensamble, picdata_projected, picclasses)
    MLJ.fit!(forest_mach, rows=traininds, verbosity=0)
    ŷ = MLJ.predict_mode(forest_mach, picdata_projected)
    time_elapsed = time() - t0

end

begin
    global results_train = DataFrame(time=Float64[], n_trees = Int[], bagging_fraction = Float64[], acc=Float64[], prec=Float64[], rec=Float64[])
    global results_test  = DataFrame(time=Float64[], n_trees = Int[], bagging_fraction = Float64[], acc=Float64[], prec=Float64[], rec=Float64[])
    for n_trees = 40:40:320
        
        for bagging_fraction in 0.5:0.1:1
            @show n_trees, bagging_fraction
            t0 = time()
            mach_pca = machine(pca(maxoutdim=15), picdata)
            MLJ.fit!(mach_pca, rows=traininds, verbosity=0)
            picdata_projected = MLJ.transform(mach_pca, picdata)

            ensamble = EnsembleModel(Tree(; max_depth=25); bagging_fraction, n=n_trees, acceleration=CPUThreads())
            forest_mach = machine(ensamble, picdata_projected, picclasses)
            MLJ.fit!(forest_mach, rows=traininds, verbosity=0)
            ŷ = MLJ.predict_mode(forest_mach, picdata_projected)
            time_elapsed = time() - t0
            # return 0
            local_result_test  = [metric()(ŷ[testinds] , picclasses[testinds])  for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
            pushfirst!(local_result_test , bagging_fraction)
            pushfirst!(local_result_test , n_trees)
            pushfirst!(local_result_test , time_elapsed)
            push!(results_test, local_result_test)


            local_result_train = [metric()(ŷ[traininds], picclasses[traininds]) for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
            pushfirst!(local_result_train, bagging_fraction)
            pushfirst!(local_result_train, n_trees)
            pushfirst!(local_result_train , time_elapsed)
            push!(results_train, local_result_train)
        end
    end
end

begin
    draw(visual(Scatter, colormap=:thermal, colorrange=extrema(results_train.acc), markersize=40) * AOG.data(results_train) * mapping(:n_trees => "Number of trees", :bagging_fraction => "Bagging fraction", color=:acc=>"Accuracy"))
    current_axis().title = "Reduced train data"; current_figure()
end

begin
    draw(visual(Scatter, colormap=:thermal, colorrange=extrema(results_test.acc), markersize=40) * AOG.data(results_test) * mapping(:n_trees => "Number of trees", :bagging_fraction => "Bagging fraction", color=:acc=>"Accuracy"))
    current_axis().title = "Reduced test data"; current_figure()
end

begin
    draw(visual(Scatter, colormap=:thermal, colorrange=extrema(results_test.acc), markersize=40) * AOG.data(filter(row->row.bagging_fraction!=1, results_test)) * mapping(:n_trees => "Number of trees", :bagging_fraction => "Bagging fraction", color=:acc=>"Accuracy"))
    current_axis().title = "Reduced test data"; current_figure()
end

begin
    @assert results_train.time == results_test.time
    draw(visual(Scatter, colormap=:thermal, markersize=40) * AOG.data(results_train) * mapping(:n => "Number of trees", :bagging_fraction => "Bagging fraction", color=:time=>"Time to train and classify (s)"))
    current_axis().title = "Reduced data timings"; current_figure()
end





begin
    global results_train_API = DataFrame(time=[], acc=[], prec=[], rec=[])
    global results_test_API  = DataFrame(time=[], acc=[], prec=[], rec=[])
    for i in 1:10
        @show n_trees, bagging_fraction
        t0 = time()
        mach_pca = machine(PCA(maxoutdim=15), picdata)
        MLJ.fit!(mach_pca, rows=traininds, verbosity=0)
        picdata_projected = MLJ.transform(mach_pca, picdata)

        ensamble = EnsembleModel(Tree(; max_depth=25); bagging_fraction, n=n_trees, acceleration=CPUThreads())
        forest_mach = machine(ensamble, picdata_projected, picclasses)
        MLJ.fit!(forest_mach, rows=traininds, verbosity=0)
        ŷ = MLJ.predict_mode(forest_mach, picdata_projected)
        time_elapsed = time() - t0
        # return 0
        local_result_test  = [metric()(ŷ[testinds] , picclasses[testinds])  for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
        # pushfirst!(local_result_test , bagging_fraction)
        # pushfirst!(local_result_test , n_trees)
        pushfirst!(local_result_test , time_elapsed)
        push!(results_test, local_result_test)


        local_result_train = [metric()(ŷ[traininds], picclasses[traininds]) for metric in [Accuracy, MulticlassPrecision, MulticlassTruePositiveRate]]
        # pushfirst!(local_result_train, bagging_fraction)
        # pushfirst!(local_result_train, n_trees)
        pushfirst!(local_result_train , time_elapsed)
        push!(results_train, local_result_train)
    end
end

eachindex(picclasses)

testinds_vec_API = batch(eachindex(pictures), 10, true)
traininds_vec_API = [deleteat!(collect(eachindex(pictures)), testinds|>sort) for testinds in testinds_vec_API]
for i in 1:10   #? Testing that it works
    @assert sort(vcat(testinds_vec_API[i], traininds_vec_API[i])) == eachindex(pictures)
end

picIDs = pictures.ID
ID_of_ind(ind) = picIDs[ind]

IDs_DJ = batch(1:33, 11, false)
using ProgressMeter
begin
    traininds_vec_DJ = Vector{Vector{Int64}}(undef, length(IDs_DJ))
    testinds_vec_DJ = Vector{Vector{Int64}}(undef, length(IDs_DJ))
    @showprogress for i in eachindex(IDs_DJ)
        bitmask = (ID_of_ind.(eachindex(pictures)) .∈ IDs_DJ[i]|>Ref)
        traininds_ = eachindex(pictures)[.!bitmask]
        testinds_ = eachindex(pictures)[bitmask]
        @assert sort(vcat(testinds_, traininds_)) == eachindex(pictures)
        @assert unique(ID_of_ind.(testinds_)) == IDs_DJ[i]
        traininds_vec_DJ[i] = traininds_
        testinds_vec_DJ[i] = testinds_
    end
end

begin
    df = DataFrame(n_trees = Int64[], μ_acc=Float64[], σ_acc=Float64[], SE_acc=[])
    for group in groupby(results_test, :n_trees)
        push!(df, [group.n_trees|>unique|>only, mean(group.acc), std(group.acc), std(group.acc)/sqrt(length(group.acc))])
    end
    df
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
GLMakie.inline!(true)
# for treedepth ∈ 2:2:10, n_PCs ∈ [5, 10, 15, 20]
using ProgressMeter


for treedepth ∈ 1:2:20
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