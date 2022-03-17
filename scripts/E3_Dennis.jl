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
using EvalMetrics
pics_33 = pictures[1:50:end]
tts_33 = TrainTestSplit(pics_33, 1//1)
nn_inds, _ = knn(tts_33, k=1)
nn_inds
trainclasses(tts_33)
classify(nn_inds, trainclasses(tts_33))

## ToDo 3.3.1 Plot the precision-recall curves for 1 to 13 “k” with “l” 
## @    values up to the “k” value. Here, the results should be one plot containing “k” lines, and each one have “k” datapoints.
function CMs(tts; k, l = 1, tiebreaker = rand, tree = BruteTree,  metric = Euclidean())
    inds, _ = knn(tts.train, tts.test; k, tree, metric)
	preds = classify(inds, trainclasses(tts); tiebreaker, l)
    truths = testclasses(tts)
    inds_missings = findall(ismissing, preds)
    n_missings  = length(inds_missings)
    preds_filtered = deleteat!(copy(preds), inds_missings) .|> identity
    truths_filtered = deleteat!(copy(truths), inds_missings)
    return [(; k, l, positive_label=i, n_missings, cm=ConfusionMatrix(OneVsRest(i, deleteat!(0:9|>Vector, i+1)), truths_filtered, preds_filtered)) for i in 0:9]
end
cms = CMs(tts_33, k=5, l=3)
cms

function CMs_summed(tts; kwargs...)
    cms = CMs(tts; kwargs...)
    summed_cm = sum(getfield.(cms, :cm))
    n_missings_summed = sum(getfield.(cms, :n_missings))
    return (;k=cms[1].k, l=cms[1].l, n_missings=n_missings_summed, summed_cm)
end

begin
    cms = []
    ks = 1:13
    for k in ks
        for l in 1:k
            push!(cms, CMs_summed(tts_33; k, l))
        end
    end
    cms
end
results = DataFrame(k = [], l = [], prec = [], rec = [], n_missings = [])
cms[1]
for cm in cms
    push!(results, [cm.k, cm.l, cm.summed_cm|>precision, cm.summed_cm|>recall, cm.n_missings])
end
results

using AlgebraOfGraphics
set_aog_theme!()
using GLMakie
begin #? plotting
    axis = (width=400, height=400)
    plt = AlgebraOfGraphics.data(results) * mapping(:prec=>"Precision", :rec=>"Recall", color=:k, marker=:l)

    draw(plt; axis)
end

#ToDo 3.3.2 Plot the maximum F1 values for each of the k in a plot together. With F1 score on the y- axis and “k”-value on the x-axis.

#ToDo 3.3.3 Discuss the results from 3.3.1 and 3.3.2. What do you think would be the most important part of a digit recognition system. Precision or recall? Please discuss in what situations would the different factors be more important?


#? 3.1: K-means clustering
#ToDo 3.1.1 Try to improve the performance on 2-person (disjunct) dataset (you can select any 2 person data for this) using K-means clustering. Perform K- means clustering of each cipher individually for the training set, in order to represent the training data as a number of cluster centroids. Now perform the training of the k-NN using the centroids of these clusters. You can try with different cluster sizes and see the resulting performance.

#ToDo 3.1.2 Compare your KNN performance based on the raw training data and based on the cluster centroids of the training data. During the comparison you should also consider the run times of the algorithm. As the generation of clusters is based on random starting points cross-validation should be performed.

#ToDo 3.1.3 Perform K-means clustering on each cipher individually for the training data from all the available datasets (disjunct). Represent the training data as a number of cluster centroids and compare performance, try multiple cluster sizes.
