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