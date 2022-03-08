using Pkg
paths = [raw"C:\Users\Dennis Bal\GeekyStuff\Julia\6. Sem\Statistical Machine Learning\drwatson", ]
for path in paths
    path|>isdir ? (path|>Pkg.activate; Pkg.instantiate(); break) : "Project not activated" |> error
end

using DrWatson

for file in readdir(srcdir())
    include(joinpath(srcdir(), file))
end

##!============================================================================!##

##! Testing produce_or_load. Save as CSV to allow saving dataframe. With JLD2,
##! only dicts can be saved.
using DataFrames

config = (k = 3, )
data, _ = produce_or_load(datadir("sims"), config, prefix = "test", suffix="csv") do config
    DataFrame(x = 10:20, k_timesx_squared= config.k.* (10:20).^2)
end
