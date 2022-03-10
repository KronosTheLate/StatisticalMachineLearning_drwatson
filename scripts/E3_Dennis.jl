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

##!======================================================!##

##?  3.2: Hierarchical clustering
##?  3.2.1 Show a low level dendrogram containing 5 instances of each digit
     #?( one person ).
using Distances

pictures = Picture.(ciphers|>eachrow) |> remove_constant
pictures_ID1 = filter(p->p.ID==1, pictures)
pictures_ID1_selection = [filter(p->p.class==i, pictures_ID1)[1:5] for i in 0:9] |> x->vcat(x...)
pictures_ID1_selection
let pics = pictures_ID1_selection
    linkage = :single  #* could be :average _complete
    n = length(pics)
    pic_dists = Matrix{Float64}(undef, n, n)
    for i in 1:n
        for j in 1:i
            the_dist = euclidean(pics[i], pics[j])
            pic_dists[i, j] = pic_dists[j, i] = the_dist
        end
    end
    global h_cluster = hclust(pic_dists; linkage)
end
h_cluster
fieldnames(Hclust)

##?  3.2.2 Use K-Means clustering to compress each digit into 5 clusters, as done in 3.1.1, and perform hierarchical clustering to show a low level dendrogram of this (one person).
##?  3.1.3 Discuss the results and relate them to the cross validation tables from k-NN classification.


function treepositions(hc::Hclust, useheight::Bool, orientation=:vertical)

    order = StatsBase.indexmap(hc.order)
    nodepos = Dict(-i => (float(order[i]), 0.0) for i in hc.order)

    xs = Array{Float64}(undef, 4, size(hc.merges, 1))
    ys = Array{Float64}(undef, 4, size(hc.merges, 1))

    for i in 1:size(hc.merges, 1)
        x1, y1 = nodepos[hc.merges[i, 1]]
        x2, y2 = nodepos[hc.merges[i, 2]]

        xpos = (x1 + x2) / 2
        ypos = useheight ?  hc.heights[i] : (max(y1, y2) + 1)
        
        nodepos[i] = (xpos, ypos)
        xs[:, i] .= [x1, x1, x2, x2]
        ys[:, i] .= [y1, ypos, ypos, y2]
    end
    if orientation == :horizontal
        return ys, xs
    else
        return xs, ys
    end
end

xs, ys = treepositions(h_cluster, true)
xs
ys


#=
3.1: K-means clustering
3.1.1 Try to improve the performance on 2-person (disjunct) dataset (you can select any 2 person data for this) using K-means clustering. Perform K- means clustering of each cipher individually for the training set, in order to represent the training data as a number of cluster centroids. Now perform the training of the k-NN using the centroids of these clusters. You can try with different cluster sizes and see the resulting performance.

3.1.2 Compare your KNN performance based on the raw training data and based on the cluster centroids of the training data. During the comparison you should also consider the run times of the algorithm. As the generation of clusters is based on random starting points cross-validation should be performed.

3.1.3 Perform K-means clustering on each cipher individually for the training data from all the available datasets (disjunct). Represent the training data as a number of cluster centroids and compare performance, try multiple cluster sizes.


3.3: Evaluation methods of k-NN
As seen in the hierarchical clustering plot we often get different labels when finding the nearest neighbors of different ciphers. This indicates that we are not completely sure about our estimation. Until now, in k-NN we have simply used the one with most votes. But we can also exclude predictions which does not have enough of the same labels. In k-NN we can set the “l” to the minimum number of “k” nearest neighbors of the strongest label to accept a match.

3.3.1 Plot the precision-recall curves for 1 to 13 “k” with “l” values up to the “k” value. Here, the results should be one plot containing “k” lines, and each one have “k” datapoints.

3.3.2 Plot the maximum F1 values for each of the k in a plot together. With F1 score on the y- axis and “k”-value on the x-axis.

3.3.3 Discuss the results from 3.3.1 and 3.3.2. What do you think would be the most important part of a digit recognition system. Precision or recall? Please discuss in what situations would the different factors be more important?

=#