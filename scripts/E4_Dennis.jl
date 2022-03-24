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

##!======================================================!##
#? Discretize the range of values of principle componten coefficient in e.g. 200


#¤ Decision trees

#¤  Compute the optimal decision point for the first 5 PCAs of a dataset (e.g. a single person) and 
#¤  compute the information gain associated to it (plot 5 graphs, one for each component, and show 
#¤  the highest information gain). See slides for how to compute information gain.

#¤  Compute a decision tree for the digit classification and visualize it. 
#¤  You can use “rpart” for creating a tree and “rpart.plot” for visualizing the tree.

#¤  Using the full data set (i.e. dataset from multiple people), evaluate a trained 
#¤  decision tree using cross validation. Try to train a tree with PCA, and without PCA 
#¤  (raw data). Discuss the important parameters.
