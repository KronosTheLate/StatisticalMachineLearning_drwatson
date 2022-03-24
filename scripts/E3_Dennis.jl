using Pkg
paths = [raw"C:\Users\Asbjo\OneDrive - Syddansk Universitet\Machine\GitHubulub\StatisticalMachineLearning_drwatson", raw"C:\Users\Dennis Bal\GeekyStuff\Julia\6. Sem\Statistical Machine Learning\drwatson"]
for path in paths
    if path|>isdir
        path|>Pkg.activate
        Pkg.instantiate()
        break
    else
        continue
    end
    "Project not activated. Ensure that the repository is
    cloned locally to one of the locations in `paths`." |> error
end

using DrWatson
for file in readdir(srcdir())
    include(joinpath(srcdir(), file))
end

if "ciphers33.RData" ∈ readdir(datadir())
    ciphers = load(datadir("ciphers33.RData"))["ciphers"]
else
    download("https://nextcloud.sdu.dk/index.php/s/Zzjcqjopy5cTawn/download/data_33.Rdata", datadir("ciphers33.RData"))
    ciphers = load(datadir("ciphers33.RData"))["ciphers"]
end

using BenchmarkHistograms
using Tools #My package
pictures = Picture.(ciphers|>eachrow) |> remove_constant |> x->sort(x, by=y->y.class)
person(ID) = filter(x -> x.ID == ID, pictures)
numbersearch(pics::Vector{<:Picture}, nr) = (filter(pic -> pic.class == nr, pics))

##!======================================================!##

#?  3.2: Hierarchical clustering
##?  3.2.1 Show a low level dendrogram containing 5 instances of each digit
#?( one person ).

import StatsPlots  ##? Import does not define stuff from StatsPlots, but makes StatsPlots.plot available. Allows simultaneous use of Makie




pictures_oneperson = person(13)
pictures_oneperson_selection = [filter(p->p.class==i, pictures_oneperson)[1:8] for i in 0:9] |> x->vcat(x...)

let pics = pictures_oneperson_selection
    linkage = :ward  #* could be :single, :average, :complete, :ward
    n = length(pics)
    pic_dists = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:i
            the_dist = euclidean(pics[i], pics[j])
            pic_dists[i, j] = pic_dists[j, i] = the_dist
        end
    end
    h_cluster = hclust(pic_dists; linkage)
    plt = StatsPlots.plot(h_cluster, title="Linkage = $linkage", size=(1920÷2, 1080÷2))
    tick_labels = map_labels(pictures_oneperson_selection.class, h_cluster)
    StatsPlots.xticks!(StatsPlots.xticks(plt)[1][1], tick_labels)
    plt
end


#! Conclusion - the hierarchical clustering does get several things right, but it is not impressive.


##?  3.2.2 Use K-Means clustering to compress each digit into 5 clusters,
#?   as done in 3.1.1, and perform hierarchical clustering to show a low 
#?   level dendrogram of this (one person).

#* Splitting pictures from 1 person into 10 vectors, each for a seperate digit.
picss_batched_322 =  batch(person(13), 10, false)
picss_batched_322[2]
#* Finding 5 k-means clusters  for each
n_clusters = 3
clusters = [kmeans(picss_batched_322[i]|>datamat, n_clusters) for i in 1:10]
clusters
#* Putting the centers of the clusters into the same matrix
centers = hcat(getfield.(clusters, :centers)...)

#* Computing the hclusters, plotting the dendrogram
let
    linkage = :ward  #* could be :single, :average, :complete, :ward
    n = size(centers, 2)
    dists = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:i
            the_dist = euclidean(centers[:, i], centers[:, j])
            dists[i, j] = dists[j, i] = the_dist
        end
    end
    h_cluster_322 = hclust(dists; linkage)
    plt = StatsPlots.plot(h_cluster_322, title="Linkage = $linkage", size=(1920÷2, 1080÷2), label="", legend=true)
    tick_labels = map_labels(repeat(0:9, n_clusters) |> sort, h_cluster_322)
    StatsPlots.xticks!(h_cluster_322.order|>eachindex, tick_labels)
    # StatsPlots.hline!([+(reverse(h_cluster_322.heights)[9:10]...)/2], label="Split in 10")
    plt
end



##?  3.2.3 Discuss the results and relate them to the cross validation
#?   tables from k-NN classification.
#* It looks like the clustering is pretty poor - most numbers are not put into the sensible clusters
#* From what I can see, clustering should DRASTICALLY make performance worse, for a small n_clusters.


#? 3.3: Evaluation methods of k-NN
#ToDo As seen in the hierarchical clustering plot we often get different labels when finding the nearest neighbors of different ciphers. This indicates that we are not completely sure about our estimation. Until now, in k-NN we have simply used the one with most votes. But we can also exclude predictions which does not have enough of the same labels. In k-NN we can set the “l” to the minimum number of “k” nearest neighbors of the strongest label to accept a match.

## ToDo 3.3.1 Plot the precision-recall curves for 1 to 13 “k” with “l” 
#  @    values up to the “k” value. Here, the results should be one plot containing “k” lines, and each one have “k” datapoints.
using PrettyTables, ProgressMeter
using OffsetArrays

function confmat(tts::TrainTestSplit; kwargs...)
    preds = classify(tts; kwargs...)
    truths = testclasses(tts)
    inds_missings = findall(ismissing, preds)
    missing_counts = OffsetArray(fill(0, 10), 0:9)
    for i in inds_missings
        missing_counts[truths[i]] += 1
    end
    deleteat!(preds, inds_missings)
    deleteat!(truths, inds_missings)
    confusion_matrix = fill(0, (10, 10))
    for i in eachindex(truths)
        confusion_matrix[truths[i]+1, preds[i]+1] += 1
    end
    return confusion_matrix, missing_counts
end

function print_confmat(cm::AbstractMatrix)
    cm = Matrix{Union{String, Int}}(cm)
    cm = [cm; sum(cm, dims=1)]
    cm = [cm sum(cm, dims=2)]
    cm = [[["$i" for i in 0:9]; "Sum"] cm]
    cm = [["          Actual\nPredicted  " reshape(["$i" for i in 0:9], (1, 10)) "Sum"]; cm]
    pretty_table(cm, noheader=true, alignment=:c, body_hlines=[1, 11], linebreaks=true)
end

begin
    params = (step = 1, parts_train=1, parts_test=1)
    results_33, results_33_path = produce_or_load(datadir(), params, prefix="33", suffix="jld2") do params #* params is the 2nd argument NamedTuple, in this case (k=3,)
        pics = pictures[1:params.step:end]
        tts = TrainTestSplit(pics, params.parts_train//params.parts_test)
        ks = 1:13
        results = DataFrame(k=Int[], l=Int[], cm=Matrix[], missing_counts = OffsetArray[])
        p = Progress(91, 1)
        for k in ks
            for l in 1:k
                cm, missing_counts = confmat(tts; k, l)
                push!(results, [k, l, cm, missing_counts])
                next!(p)
            end
        end
        return @strdict results
    end
    results_33 = results_33["results"]
end

function accuracy(cm::AbstractMatrix)
    @assert size(cm) == (10, 10) "Expected a 10x10 confusion matric"
    sum(cm[i, i] for i in 1:10) / sum(cm, dims=(1, 2))[1]
end

using AlgebraOfGraphics
begin
    axis = (width=400, height=400)
    plt = visual(Scatter, colormap=:thermal) * AlgebraOfGraphics.data(results_33) * mapping(:k, :cm=>accuracy=>"Accuracy")
    plt *= mapping(color = :l => "Threshold l")
    draw(plt; axis)
    current_axis().xticks = 1:13
    current_axis().title = "$(length(pictures)÷params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split"
    current_figure()
end
begin #? Same plot as above with proportion of classifications made on colorbar
    axis = (width=400, height=400)
    plt = visual(Scatter, colormap=:thermal) * AlgebraOfGraphics.data(results_33) * mapping(:k, :cm=>accuracy=>"Accuracy")
    plt *= mapping(color = :missing_counts => (x->(length(pictures)/params.step * (params.parts_test / (params.parts_train + params.parts_test)) - sum(x))/(length(pictures)/params.step * (params.parts_test / (params.parts_train + params.parts_test)))) => "Proportion of points classified")
    draw(plt; axis)
    current_axis().xticks = 1:13
    current_axis().title = "$(length(pictures)÷params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split"
    current_figure()
end
scatter(results_33.missing_counts .|> sum, results_33.cm .|> accuracy,
axis=(xlabel="Missing count", ylabel="Accuracy", title = "$(length(pictures)÷params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split"))
begin
    axis = (width=400, height=400)
    plt = visual(Scatter, colormap=:thermal) * AlgebraOfGraphics.data(results_33) * mapping(:k, :l=>"Threshold l")
    plt *= mapping(color = :cm => accuracy => "Accuracy")
    draw(plt; axis)
    current_axis().xticks = 1:13
    current_axis().yticks = 1:13
    current_axis().title = "$(length(pictures)÷params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split"
    current_figure()
end

begin
    axis = (width=400, height=400)
    plt = visual(Heatmap, colormap=:thermal) * AlgebraOfGraphics.data(results_33) * mapping(:k, :l=>"Threshold l", :cm=>accuracy=>"Accuracy")
    draw(plt; axis)
    current_axis().xticks = 1:13
    current_axis().yticks = 1:13
    current_axis().title = "$(length(pictures)÷params.step) pictures in total, $(params.parts_train)/$(params.parts_test) split"
    current_figure()
end

##? Ignoring missings leads to following definition of prec and recall
import Base: precision
function precision(cm::Matrix)
    @assert size(cm) == (10, 10) "Expected a 10x10 matrix"
    [cm[i, i]/sum(cm[j, i] for j in 1:10) for i in 1:10]
end
avg_precision(cm::Matrix) = mean(filter(!isnan, precision(cm)))

function recall(cm::Matrix)
    @assert size(cm) == (10, 10) "Expected a 10x10 matrix"
    [cm[i, i]/sum(cm[i, j] for j in 1:10) for i in 1:10]
end
avg_recall(cm::Matrix) = mean(filter(!isnan, recall(cm)))
#!Mention NaN's, coming from no real DIGIT and no predicted DIGIT.

F1_score(cm::Matrix) = 2(avg_precision(cm) ⊕ avg_recall(cm))

let df = DataFrame((cm = results_33.cm, k=results_33.k, l=results_33.l, prec=avg_precision.(results_33.cm), rec=avg_recall.(results_33.cm), F1=F1_score.(results_33.cm)))
    my_data = groupby(select(df, [:cm, :k]), :k)
    max_f1s = []
    for group in my_data
        F1s_current_group = []
        for cm in group.cm
            push!(F1s_current_group, F1_score(cm))
        end
        push!(max_f1s, maximum(F1s_current_group))
    end
    max_f1s .|> identity
    result = DataFrame(max_F1=identity.(max_f1s), k=1:13)
    scatter(result.k, result.max_F1, axis=(xlabel="k", ylabel="Maximum F1 score", xticks=1:13))
    # plt = AlgebraOfGraphics.data(result) * mapping(:k, :max_F1)
    # draw(plt)
end


let
    my_data = (k=results_33.k, l=results_33.l, prec=avg_precision.(results_33.cm), rec=avg_recall.(results_33.cm), F1=F1_score.(results_33.cm))
    # @show my_data.l
    plt = visual(Scatter, colormap=:thermal) * AlgebraOfGraphics.data(my_data)
    
    plt1 = plt * mapping(:rec=>"Recall", :prec=>"Precision")
    plt1 *= mapping(color=:l=>"Threshold l")

    plt2 = plt * macpping(:k=>"k", :F1=>"F1 score")
    plt2 *= mapping(color=:l=>"Threshold l")

    plt3 = plt * mapping(:k=>"k", :l=>"Threshold l")
    plt3 *= mapping(color=:F1=>"F1 score")
    fig = draw(plt1)

    # current_figure()
end

#ToDo 3.3.2 Plot the maximum F1 values for each of the k in a plot together. With F1 score on the y- axis and “k”-value on the x-axis.


###! Below is the old code that gave a straigt line.
##
function CMs(tts; k, l = 1, tiebreaker = rand, tree = BruteTree,  metric = Euclidean())
    inds = knn_threaded(tts.train, tts.test; k, tree, metric)
	preds = classify(inds, trainclasses(tts); tiebreaker, l)
    truths = testclasses(tts)
    inds_missings = findall(ismissing, preds)
    n_missings  = length(inds_missings)
    preds_filtered = deleteat!(copy(preds), inds_missings) .|> identity
    truths_filtered = deleteat!(copy(truths), inds_missings)
    return [(; k, l, positive_label=i, n_missings, cm=ConfusionMatrix(OneVsRest(i, deleteat!(0:9|>Vector, i+1)), truths_filtered, preds_filtered)) for i in 0:9]
end

function CMs_summed(tts; kwargs...)
    cms = CMs(tts; kwargs...)
    summed_cm = sum(getfield.(cms, :cm))
    n_missings_summed = sum(getfield.(cms, :n_missings))
    return (;k=cms[1].k, l=cms[1].l, n_missings=n_missings_summed, summed_cm)
end

#=
@time begin
    cms = []
    ks = 1:13
    t₀ = time()
    for k in ks
        for l in 1:k
            "k=$k, l=$l. Time elapsed: $(round(time() - t₀), digits=1) seconds" |> println
            push!(cms, CMs_summed(tts_33; k, l))
        end
    end
    cms
end
=#
# save(datadir("cms_alldata_TTSRatio_99_1.jld2"), Dict("cms"=>cms, ))
cms = load(datadir("cms_alldata_TTSRatio_99_1.jld2"))["cms"]

##

begin
    results = DataFrame(k = [], l = [], prec = [], rec = [], n_missings = [], f1 = [])
    for cm in cms
        push!(results, [cm.k, cm.l .|> string, cm.summed_cm|>precision, cm.summed_cm|>recall, cm.n_missings, cm.summed_cm|>f1_score])
    end
    results .|> identity
end
cms
getfield.(cms, :summed_cm)
begin  #? Plotting setup
    using AlgebraOfGraphics
    set_aog_theme!()
    update_theme!(markersize=15, fontsize=30)
    using GLMakie
end

begin #? plotting
    axis = (width=900, height=800)
    plt = AlgebraOfGraphics.data(results) * mapping(:prec=>"Precision", :rec=>"Recall")
    plt *= mapping(color=:k)
    plt *= mapping(marker=:l=>sorter(Vector(1:13) .|> string))

    draw(plt; axis, colorbar=(colormap=:thermal, ))
end

#ToDo 3.3.2 Plot the maximum F1 values for each of the k in a plot together. With F1 score on the y- axis and “k”-value on the x-axis.
begin
    new_results = results|>copy
    sort!(new_results, :f1)
    bad_inds = Int64[]
    represented_ks = Int64[]
    for i in axes(new_results, 1)
        if new_results[i, :k] ∈ represented_ks
            push!(bad_inds, i)
        else
            push!(represented_ks, new_results[i, :k])
        end
    end
    new_results = new_results[axes(new_results, 1) .∉ [bad_inds], :] .|> identity
    begin #? plotting
        axis = (width=900, height=800)
        plt = AlgebraOfGraphics.data(new_results) * mapping(:k=>"k", :f1=>"Maximal F1 score")
        plt *= mapping(color=:l=>sorter(Vector(1:13) .|> string))
    
        draw(plt; axis)
    end
    current_axis().xticks = 1:13
    current_figure()
end

#ToDo 3.3.3 Discuss the results from 3.3.1 and 3.3.2. What do you think would be the most important part of a digit recognition system. Precision or recall? Please discuss in what situations would the different factors be more important?


#? 3.1: K-means clustering
#ToDo 3.1.1 Try to improve the performance on 2-person (disjunct) dataset (you can select any 2 person data for this) using K-means clustering. Perform K- means clustering of each cipher individually for the training set, in order to represent the training data as a number of cluster centroids. Now perform the training of the k-NN using the centroids of these clusters. You can try with different cluster sizes and see the resulting performance.

#ToDo 3.1.2 Compare your KNN performance based on the raw training data and based on the cluster centroids of the training data. During the comparison you should also consider the run times of the algorithm. As the generation of clusters is based on random starting points cross-validation should be performed.

#ToDo 3.1.3 Perform K-means clustering on each cipher individually for the training data from all the available datasets (disjunct). Represent the training data as a number of cluster centroids and compare performance, try multiple cluster sizes.
