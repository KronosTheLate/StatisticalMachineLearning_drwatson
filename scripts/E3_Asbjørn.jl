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


if "ciphers33.RData" ∈ readdir(datadir())
  ciphers = load(datadir("ciphers33.RData"))["ciphers"]
else
  download("https://nextcloud.sdu.dk/index.php/s/Zzjcqjopy5cTawn/download/data_33.Rdata", datadir("ciphers33.RData"))
  ciphers = load(datadir("ciphers33.RData"))["ciphers"]
end





