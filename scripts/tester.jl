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
using DataFrames


config = (k = 3, )
data1, _ = produce_or_load(datadir("sims"), config, prefix = "test1", suffix="csv") do config
    DataFrame(x = 10:20, k_timesx_squared= config.k.* (10:20).^2)
end
data1
data2, _ = produce_or_load(datadir("sims"), config, prefix = "test2", suffix="csv") do config
    return DataFrame(x = 10:20, k_timesx_squared= config.k.* (10:20).^2)
end
data2
data3, _ = produce_or_load(datadir("sims"), config, prefix = "test3", suffix="csv") do config
    results = DataFrame(x = 10:20, k_timesx_squared= config.k.* (10:20).^2)
    return results
end
data3