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


##!============================================================================!##

##! SVectors

using NearestNeighbors
pic = pictures[1].data
pics = pictures[1:100].data
tree = BruteTree(hcat(pics...))

const datalength = length(pic)

spic = @SVector [pic[i] for i in 1:datalength]
spics = [@SVector [pic[i] for i in 1:datalength] for pic in pics]
stree = BruteTree(spics)


Threads.@threads for i in 1:8
    @show Threads.threadid()
    @show i
end

for i in 1:8
    @show Threads.threadid()
    @show i
end

trainpics = rand(pictures, 10000)
testpics = rand(pictures, 1000)
tree = BruteTree(trainpics|>datamat)
@time knn(tree, testpics|>datamat, 3)

@time knn_threaded(trainpics, testpics; k=3)
@time knn(trainpics, testpics; k=3)



##! Testing produce_or_load. Save as CSV to allow saving dataframe. With JLD2,
##! only dicts can be saved.
using DataFrames
#*                                         parameters   filename before parameters  filename after parameters 
data, _ = produce_or_load(datadir("sims", "subfolder"), (k = 3, length=20),   prefix = "test_run",            suffix="csv"             ) do params #* params is the 2nd argument NamedTuple, in this case (k=3,)
    DataFrame(x = 10:20, k_timesx_squared= params.k.* (10:20).^2)
end