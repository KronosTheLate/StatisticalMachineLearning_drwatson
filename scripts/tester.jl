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


##!============================================================================!##

##! Testing produce_or_load. Save as CSV to allow saving dataframe. With JLD2,
##! only dicts can be saved.
using DataFrames
#*                                         parameters   filename before parameters  filename after parameters 
data, _ = produce_or_load(datadir("sims", "subfolder"), (k = 3, length=20),   prefix = "test_run",            suffix="csv"             ) do params #* params is the 2nd argument NamedTuple, in this case (k=3,)
    DataFrame(x = 10:20, k_timesx_squared= params.k.* (10:20).^2)
end

