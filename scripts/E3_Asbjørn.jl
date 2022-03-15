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
pictures = Picture.( ciphers|>eachrow );


# lets find 2 people
#! Comment - calling `remove constant` in the function `person`
#! is bad practice - the name `person` does not suggest that the
#! constant is removed. Maybie remove the constant values when defining the data?
person(ID) = filter(x -> x.ID == ID, pictures) |>remove_constant # returns ID, CLass, and data for person 1
numbersearch(data, nr) = (filter(x -> x.class == nr, data))



xData = numbersearch(person(12),3)|>datamat  

# cluster X into 20 clusters using K-means
R = kmeans(xData, 10; maxiter=200);

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

##? 

function GetClusterCenter(data::Matrix, numberofcluster::Int64) 
M = (kmeans(data, numberofcluster; maxiter=200)).centers
return M
end


#! finder M cluser for person 2 for 0 til 9
Clustersize = 1;
temp = []
for i in 0:9
M=GetClusterCenter( numbersearch( person(2), i )|>datamat ,Clustersize);
push!(temp,M)
end
temp
#! Sætter dem sammen i en picture sturct

tempfirst = hcat(repeat(2:2,Clustersize), repeat(0:0,Clustersize), temp[1]')
for i in 0:9
temp2 = hcat(repeat(2:2,Clustersize), repeat(i:i,Clustersize), temp[1+i]')
tempfirst = vcat(tempfirst, temp2)
end
Mpictures = Picture.(tempfirst |> eachrow)


#! test for accuracy 

function TestAcc(traindata,testdata, k)
inds, _ = knn(traindata, testdata; k=k, tree=BruteTree, metric=Euclidean());
preds = classify(inds, traindata.class);
acc = Statistics.mean(preds .== testdata.class)
return acc
end



TestAcc(person(2), person(5), 1)

##?
"""
    cluster_center_matrix(Clustersize::Int, personid::Int)

Builds the center cluster matrix of a single person where numbers 0-9 is clusterd.
"""
function cluster_center_matrix(Clustersize::Int, personid::Int) 

    #! finder M cluster for person 2 for 0 til 9
    let
    temp = []
    for i in 0:9
    M=GetClusterCenter( numbersearch( person(personid), i )|>datamat ,Clustersize);
    push!(temp,M)
    end
    temp
    #! Sætter dem sammen i en picture sturct
    
    tempfirst = hcat(repeat(2:2,Clustersize), repeat(0:0,Clustersize), temp[1]')
    for i in 0:9
    temp2 = hcat(repeat(2:2,Clustersize), repeat(i:i,Clustersize), temp[1+i]')
    tempfirst = vcat(tempfirst, temp2)
    end
    Mpictures = Picture.(tempfirst |> eachrow)
    return Mpictures
    end
end


##? 
#! This is the ploting place ! 

fig = Figure()
ax = Axis(fig[1,1])
sca20 = scatter!([1], [TestAcc(cluster_center_matrix(20, 2), person(5), 1)], color=:blue)
sca10 = scatter!([1], [TestAcc(cluster_center_matrix(10, 2), person(5), 1)],color=:red)
sca5 = scatter!([1], [TestAcc(cluster_center_matrix(5, 2), person(5), 1)],color=:green)
scaOG =scatter!([1], [TestAcc(person(2), person(5), 1)], color=:black)

scatter!([2], [TestAcc(cluster_center_matrix(20, 12), person(13), 1)], label = "20 cluster", color=:blue)
scatter!([2], [TestAcc(cluster_center_matrix(10, 12), person(13), 1)], label = "10 cluster", color=:red)
scatter!([2], [TestAcc(cluster_center_matrix(5, 12), person(13), 1)], label = "5 cluster", color=:green)
scatter!([2], [TestAcc(person(12), person(13), 1)], label = "KNN no clustering", color=:black)

scatter!([3], [TestAcc(cluster_center_matrix(20, 4), person(8), 1)], label = "20 cluster", color=:blue)
scatter!([3], [TestAcc(cluster_center_matrix(10, 4), person(8), 1)], label = "10 cluster", color=:red)
scatter!([3], [TestAcc(cluster_center_matrix(5, 4), person(8), 1)], label = "5 cluster", color=:green)
scatter!([3], [TestAcc(person(4), person(8), 1)], label = "KNN no clustering", color=:black)


scatter!([4], [TestAcc(cluster_center_matrix(20, 20), person(25), 1)], label = "20 cluster", color=:blue)
scatter!([4], [TestAcc(cluster_center_matrix(10, 20), person(25), 1)], label = "10 cluster", color=:red)
scatter!([4], [TestAcc(cluster_center_matrix(5, 20), person(25), 1)], label = "5 cluster", color=:green)
scatter!([4], [TestAcc(person(20), person(25), 1)], label = "KNN no clustering", color=:black)


Legend(fig[1, 2],
    [sca20, sca10, sca5, scaOG],
    ["20 clusters", "10 clusters", "5 clusters", "KNN no clustering"])

processors = ["2 vs. 1", "12 vs. 13", "4 vs. 8", "20 vs. 25"]
#sca20.ylabel = "Hello"
ax.xticks = (1:4, processors)
fig

#! seems highly dependet on the people you chose to compare as to which is better... strange. 

##? Cross validation

 function crossvald(clustersize, clusterperson, testperson, k, rep)
accSaver = []
for i in 1:10
push!(accSaver, TestAcc(cluster_center_matrix(clustersize, clusterperson), testperson, k);)
end
return accSaver
end
accSaver |> mean
accSaver |> var |> √

cl20v21 = crossvald(20, 20 , person(21), 1, 1:50)
cl12v13 = crossvald(20, 12 , person(13), 1, 1:50)
cl1v2 = crossvald(20, 1 , person(2), 1, 1:10)

cl20v21_10 = crossvald(10, 20 , person(21), 1, 1:10)
cl12v13_10 = crossvald(10, 12 , person(13), 1, 1:10)
cl1v2_10 = crossvald(10, 1 , person(2), 1, 1:10)




fig = Figure()

ax=Axis(fig[1,1])

xs = 1:1:3
ys1 = [mean(cl20v21), mean(cl12v13), mean(cl1v2)]
#ys2 = [mean(cl20v21_10), mean(cl12v13_10), mean(cl1v2_10)]

lowerrors = [var(cl20v21)|>√, var(cl12v13)|>√, var(cl1v2)|>√]
higherrors = [var(cl20v21)|>√, var(cl12v13)|>√, var(cl1v2)|>√]

#lowerrors_10 = [var(cl20v21_10)|>√, var(cl12v13_10)|>√, var(cl1v2_10)|>√]
#higherrors_10 = [var(cl20v21_10)|>√, var(cl12v13_10)|>√, var(cl1v2_10)|>√]


errorbars!(xs, ys1, lowerrors, higherrors,
    color = :green,
    whiskerwidth = 10)

    
#errorbars!(xs, ys2, lowerrors_10, higherrors_10,
#color = :blue,
#whiskerwidth = 10)

# plot position scatters so low and high errors can be discriminated
sca = scatter!(xs, ys1, color = :green)
#scatter!(xs, ys2, color = :blue)
scaOG = scatter!(xs, [TestAcc(person(20), person(21), 1), TestAcc(person(12), person(13), 1), TestAcc(person(1), person(2), 1)],
    color = :black
)

processors = ["20 vs. 21", "12 vs. 13", "1 vs. 2"]
#sca20.ylabel = "Hello"
ax.xticks = (1:3, processors)

Legend(fig[1, 2],
    [sca, scaOG],
    ["20 Clusters", "KNN no clustering"])


fig


##? 

##! alt data! 



tts = TrainTestSplit(pictures |> remove_constant, 1//1, false) 


"""
    cluster_center_matrix(Clustersize::Int, personid::Int)

Builds the center cluster matrix of a single person where numbers 0-9 is clusterd.
"""
function cluster_center_matrixAll(Clustersize::Int, data::Vector{Picture{Float64}}) 

    #! finder M cluster for person 2 for 0 til 9
    let
    temp = []
    for i in 0:9
    M=GetClusterCenter( numbersearch( data, i )|>datamat ,Clustersize);
    push!(temp,M)
    end
    temp
    #! Sætter dem sammen i en picture sturct
    
    tempfirst = hcat(repeat(69:69,Clustersize), repeat(0:0,Clustersize), temp[1]')
    for i in 1:9
    temp2 = hcat(repeat(69:69,Clustersize), repeat(i:i,Clustersize), temp[1+i]')
    tempfirst = vcat(tempfirst, temp2)
    end
    Mpictures = Picture.(tempfirst |> eachrow)
    return Mpictures
    end
end

Mpictures = cluster_center_matrixAll(300, tts.train)

TestAcc(Mpictures, tts.test, 1)
TestAcc(tts.train, tts.test, 1)

Mpictures[91] |> visualize_picture
