using Pkg
paths = [raw"C:\Users\Dennis Bal\GeekyStuff\Julia\6. Sem\Statistical Machine Learning\drwatson", raw"C:\Users\Asbjo\OneDrive - Syddansk Universitet\Machine\GitHubulub\StatisticalMachineLearning_drwatson" ]
for path in paths
    if path|>isdir
        path|>Pkg.activate
        Pkg.instantiate()
    end
#    "Project not activated" |> error
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