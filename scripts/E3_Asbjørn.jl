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


#= 
3.1.1 Try to improve the performance on 2-person (disjunct) 
dataset (you can select any 2 person data for this) using K-means clustering.
 Perform K- means clustering of each cipher individually for the training set, in order to 
 represent the training data as a number of cluster centroids. Now perform the training of 
 the k-NN using the centroids
 of these clusters. You can try with different cluster sizes and see the resulting performance.
=# 


# This defines the data set in a structure, call pictures.ID for id,
# pictures.class for class
# pictures.data for data of pictures
ciphers
pictures = Picture.( ciphers|>eachrow )


# lets find 2 people
#! Comment - calling `remove constant` in the function `person`
#! is bad practice - the name `person` does not suggest that the
#! constant is removed. Maybie remove the constant values when defining the data?
person(ID) = filter(x -> x.ID == ID, pictures) |>remove_constant # returns ID, CLass, and data for person 1
numbersearch(data, nr) = (filter(x -> x.class == nr, data))




xData = numbersearch(person(12),3)|>datamat  

# cluster X into 20 clusters using K-means
R = kmeans(xData, 10; maxiter=200, display=:iter);

a = assignments(R) # get the assignments of points to clusters
c = counts(R) # get the cluster sizes
M = R.centers # get the cluster centers


colormap = :flag_no

fig = Figure()
heatmap(fig[1,1],unflatten( M[:,1] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "1, number of occurences $(c[1])",))
heatmap(fig[1,2],unflatten( M[:,2] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "2 number of occurences $(c[2])",))
heatmap(fig[1,3],unflatten( M[:,3] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "3 number of occurences $(c[3])",))
heatmap(fig[1,4],unflatten( M[:,4] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "4 number of occurences $(c[4])",))
heatmap(fig[1,5],unflatten( M[:,5] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "5 number of occurences $(c[5])",))
heatmap(fig[2,1],unflatten( M[:,6] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "6, number of occurences $(c[6])",))
heatmap(fig[2,2],unflatten( M[:,7] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "7, number of occurences $(c[7])",))
heatmap(fig[2,3],unflatten( M[:,8] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "8, number of occurences $(c[8])",))
heatmap(fig[2,4],unflatten( M[:,9] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "9, number of occurences $(c[9])",))
heatmap(fig[2,5],unflatten( M[:,10] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "10, number of occurences $(c[10])",))

#fieldnames(Figure)
# cb=Colorbar(fig[1:3, 7], limits=extrema(billede1); colormap )
hidedecorations!.(fig.content[begin:end])
fig

