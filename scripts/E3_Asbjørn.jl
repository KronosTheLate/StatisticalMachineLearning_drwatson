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


colormap = :flag_ua

fig = Figure()
heatmap(fig[1,1],unflatten( M[:,1] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 1, # occurences $(c[1])",))
heatmap(fig[1,2],unflatten( M[:,2] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 2, # occurences $(c[2])",))
heatmap(fig[1,3],unflatten( M[:,3] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 3, # occurences$(c[3])",))
heatmap(fig[1,4],unflatten( M[:,4] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 4, # occurences $(c[4])",))
heatmap(fig[1,5],unflatten( M[:,5] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 5, # occurences $(c[5])",))
heatmap(fig[2,1],unflatten( M[:,6] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 6, # occurences $(c[6])",))
heatmap(fig[2,2],unflatten( M[:,7] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 7, # occurences $(c[7])",))
heatmap(fig[2,3],unflatten( M[:,8] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 8, # occurences $(c[8])",))
heatmap(fig[2,4],unflatten( M[:,9] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 9, # occurences $(c[9])",))
heatmap(fig[2,5],unflatten( M[:,10] )|>reverse|>x->reverse(x, dims=1); colormap, axis = (title = "cluster center 10, # occurences $(c[10])",))

#fieldnames(Figure)
# cb=Colorbar(fig[1:3, 7], limits=extrema(billede1); colormap )
hidedecorations!.(fig.content[begin:end])
fig

##? 

function GetClusterCenter(data::Matrix, numberofcluster::Int64) 
M = (kmeans(data, numberofcluster; maxiter=500)).centers
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
ax2  = Axis(fig[2,1])
sca20T = scatter!([1], [TestAcc(cluster_center_matrix(20, 2), person(2), 1)], color=:blue, marker = :rect)
sca10T = scatter!([1], [TestAcc(cluster_center_matrix(10, 2), person(2), 1)],color=:red, marker = :rect)
sca5T = scatter!([1], [TestAcc(cluster_center_matrix(5, 2), person(2), 1)],color=:green, marker = :rect)
scaOGT =scatter!([1], [TestAcc(person(2), person(2), 1)], color=:black, marker = :rect)

scatter!(ax, [2], [TestAcc(cluster_center_matrix(20, 12), person(13), 1)], label = "20 cluster", color=:blue)
scatter!(ax, [2], [TestAcc(cluster_center_matrix(10, 12), person(13), 1)], label = "10 cluster", color=:red)
scatter!(ax, [2], [TestAcc(cluster_center_matrix(5, 12), person(13), 1)], label = "5 cluster", color=:green)
scatter!(ax, [2], [TestAcc(person(12), person(13), 1)], label = "KNN no clustering", color=:black)
scatter!(ax2, [2], [TestAcc(cluster_center_matrix(20, 12), person(12), 1)], color=:blue, marker = :rect)
scatter!(ax2, [2], [TestAcc(cluster_center_matrix(10, 12), person(12), 1)],color=:red, marker = :rect)
scatter!(ax2, [2], [TestAcc(cluster_center_matrix(5, 12), person(12), 1)],color=:green, marker = :rect)
scatter!(ax2, [2], [TestAcc(person(12), person(12), 1)], color=:black, marker = :rect)

scatter!(ax, [3], [TestAcc(cluster_center_matrix(20, 4), person(8), 1)], label = "20 cluster", color=:blue)
scatter!(ax, [3], [TestAcc(cluster_center_matrix(10, 4), person(8), 1)], label = "10 cluster", color=:red)
scatter!(ax, [3], [TestAcc(cluster_center_matrix(5, 4), person(8), 1)], label = "5 cluster", color=:green)
scatter!(ax, [3], [TestAcc(person(4), person(8), 1)], label = "KNN no clustering", color=:black)
scatter!(ax2, [3], [TestAcc(cluster_center_matrix(20, 4), person(4), 1)], color=:blue, marker = :rect)
scatter!(ax2, [3], [TestAcc(cluster_center_matrix(10, 4), person(4), 1)],color=:red, marker = :rect)
scatter!(ax2, [3], [TestAcc(cluster_center_matrix(5, 4), person(4), 1)],color=:green, marker = :rect)
scatter!(ax2, [3], [TestAcc(person(4), person(4), 1)], color=:black, marker = :rect)

scatter!(ax, [4], [TestAcc(cluster_center_matrix(20, 20), person(25), 1)], label = "20 cluster", color=:blue)
scatter!(ax, [4], [TestAcc(cluster_center_matrix(10, 20), person(25), 1)], label = "10 cluster", color=:red)
scatter!(ax, [4], [TestAcc(cluster_center_matrix(5, 20), person(25), 1)], label = "5 cluster", color=:green)
scatter!(ax, [4], [TestAcc(person(20), person(25), 1)], label = "KNN no clustering", color=:black)
scatter!(ax2, [4], [TestAcc(cluster_center_matrix(20, 20), person(20), 1)], color=:blue, marker = :rect)
scatter!(ax2, [4], [TestAcc(cluster_center_matrix(10, 20), person(20), 1)],color=:red, marker = :rect)
scatter!(ax2, [4], [TestAcc(cluster_center_matrix(5, 20), person(20), 1)],color=:green, marker = :rect)
scatter!(ax2, [4], [TestAcc(person(20), person(20), 1)], color=:black, marker = :rect)


# Legend(fig[1:2, 2],
#    [sca20, sca10, sca5, scaOG, sca20T, sca10T, sca5T, scaOGT],
#    ["20 clusters test", "10 clusters test", "5 clusters test", "KNN no clustering test", "20 clusters train", "10 clusters train", "5 clusters train", "KNN no clustering train"],
#    "Trig Functions",
#    nbanks = 1
#    )

groupcolor = [sca20, sca10, sca5, scaOG];
groupshape = [scaOG, scaOGT]

legends = [Legend(fig[1:2,2],
[groupcolor, groupshape],
[["20", "10", "5", "NaN"], ["Test", "Train"]],
["Number of clusters", "Train or Test"]) for _ in 1:7]

processors = ["case 1", "case 2", "case 3", "case 4"]
#sca20.ylabel = "Hello"
ax2.xticks = (1:4, processors)
hidexdecorations!(ax, grid = false)
Label(fig[1:2,0], "Accuracy for KNN",rotation = π/2)
fig

#! seems highly dependet on the people you chose to compare as to which is better... strange. 

##? Cross validation

 function crossvald(clustersize, clusterperson, testperson, k, rep)
accSaver = []
for i in rep
push!(accSaver, TestAcc(cluster_center_matrix(clustersize, clusterperson), testperson, k);)
end
return accSaver
end


cl20v21 = crossvald(20, 2 , person(5), 1, 1:10)
cl12v13 = crossvald(20, 12 , person(13), 1, 1:10)
cl1v2 = crossvald(20, 4 , person(8), 1, 1:10)

cl20v21_10 = crossvald(10, 2 , person(5), 1, 1:10)
cl12v13_10 = crossvald(10, 12 , person(13), 1, 1:10)
cl1v2_10 = crossvald(10, 4 , person(8), 1, 1:10)




fig = Figure()

ax=Axis(fig[1,1])

xs = 1:1:3
ys1 = [mean(cl20v21), mean(cl12v13), mean(cl1v2)]
ys2 = [mean(cl20v21_10), mean(cl12v13_10), mean(cl1v2_10)]

lowerrors = [var(cl20v21)|>√, var(cl12v13)|>√, var(cl1v2)|>√]
higherrors = [var(cl20v21)|>√, var(cl12v13)|>√, var(cl1v2)|>√]

lowerrors_10 = [var(cl20v21_10)|>√, var(cl12v13_10)|>√, var(cl1v2_10)|>√]
higherrors_10 = [var(cl20v21_10)|>√, var(cl12v13_10)|>√, var(cl1v2_10)|>√]


errorbars!(collect(xs) .-0.1, ys1, lowerrors, higherrors,
    color = :blue,
    whiskerwidth = 10)

    
errorbars!(collect(xs) .+0.1, ys2, lowerrors_10, higherrors_10,
color = :red,
whiskerwidth = 10)

# plot position scatters so low and high errors can be discriminated
sca20 = scatter!(collect(xs) .-0.1, ys1, color = :blue)
sca10 =scatter!(collect(xs) .+0.1, ys2, color = :red)
scaOG = scatter!(xs, [TestAcc(person(2), person(5), 1), TestAcc(person(12), person(13), 1), TestAcc(person(4), person(8), 1)],
    color = :black
)

processors = ["Case 1", "Case 2", "Case 3"]
#sca20.ylabel = "Hello"
ax.xticks = (1:3, processors)

Legend(fig[1, 2],
    [sca20, sca10, scaOG],
    ["20 Clusters", "10 Clusters","NaN"])

Label(fig[1,0], "Accuracy for KNN",rotation = π/2, tellheight = false)

fig


##?  time functions ----------------------------------------------------------
using TimerOutputs

# case 1 for 20 clusters

Batched_person12 = [ numbersearch( person(12), i ) |>datamat  for i in 0:9 ];
test = person(13);
train = person(12);


# R = kmeans(Batched_person12[i], 20; maxiter=200)
# Mx = R.centers
to = TimerOutput();

Clustersize = 20
 #! finder M cluster for person X for 0 til 9
    temp = []
    @timeit to "Batching" for i in 1:10
        @timeit to "Finding 20 clusters" R = kmeans(Batched_person12[i], Clustersize; maxiter=200)
        @timeit to "Finding centers" M = R.centers
    push!(temp,M)
    end
    #! Sætter dem sammen i en picture sturct

    @timeit to "Reapplying labels" tempfirst = hcat(repeat(2:2,Clustersize), repeat(0:0,Clustersize), temp[1]')
    for i in 1:9
    temp2 = hcat(repeat(2:2,Clustersize), repeat(i:i,Clustersize), temp[1+i]')
    tempfirst = vcat(tempfirst, temp2)
    end
    @timeit to "Converting to struct" Mpictures = Picture.(tempfirst |> eachrow)


@timeit to "KNN classifying 20 clusters" TestAcc(Mpictures, test, 1)
@timeit to "KNN classifying" TestAcc(train, test, 1)
to
##?
##! alt data! -------------------------------------------------------------------------------------------



tts = TrainTestSplit(pictures[1:10:end] |> remove_constant, 1//1, false) 


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

function crossvaldALL(clustersize, Traindata, Testdata, k, rep)
    accSaver = []
    for i in rep
    push!(accSaver, TestAcc(cluster_center_matrixAll(clustersize, Traindata), Testdata, k);)
    end
    return accSaver
    end

crossvaldALL(10, tts.train, tts.test, 1, 1:5)



acc20 = crossvaldALL(20, tts.train, tts.test, 1, 1:10)
acc40 = crossvaldALL(40, tts.train, tts.test, 1, 1:10)
acc60 = crossvaldALL(60, tts.train, tts.test, 1, 1:10)
acc80 = crossvaldALL(80, tts.train, tts.test, 1, 1:10)
acc100 = crossvaldALL(100, tts.train, tts.test, 1, 1:10)
acc120 = crossvaldALL(120, tts.train, tts.test, 1, 1:10)
acc140 = crossvaldALL(140, tts.train, tts.test, 1, 1:10)
acc160 = crossvaldALL(160, tts.train, tts.test, 1, 1:10)
acc180 = crossvaldALL(180, tts.train, tts.test, 1, 1:10)
acc200 = crossvaldALL(200, tts.train, tts.test, 1, 1:10)


acc20T = crossvaldALL(20, tts.train, tts.train, 1, 1:10)
acc40T = crossvaldALL(40, tts.train, tts.train, 1, 1:10)
acc60T = crossvaldALL(60, tts.train, tts.train, 1, 1:10)
acc80T = crossvaldALL(80, tts.train, tts.train, 1, 1:10)
acc100T = crossvaldALL(100, tts.train, tts.train, 1, 1:10)
acc120T = crossvaldALL(120, tts.train, tts.train, 1, 1:10)
acc140T = crossvaldALL(140, tts.train, tts.train, 1, 1:10)
acc160T = crossvaldALL(160, tts.train, tts.train, 1, 1:10)
acc180T = crossvaldALL(180, tts.train, tts.train, 1, 1:10)
acc200T = crossvaldALL(200, tts.train, tts.train, 1, 1:10)

accOG = TestAcc(tts.train, tts.test, 1)
accOGT = TestAcc(tts.train, tts.train, 1)


##? 

errorhigh = [var(acc20)|>√, var(acc40)|>√, var(acc60)|>√,var(acc80)|>√, var(acc100)|>√, var(acc120)|>√, var(acc140)|>√,  var(acc160)|>√,  var(acc180)|>√, var(acc200)|>√]
errorlow = errorhigh;

errorhighT = [var(acc20T)|>√, var(acc40T)|>√, var(acc60T)|>√,var(acc80T)|>√, var(acc100T)|>√, var(acc120T)|>√, var(acc140T)|>√,  var(acc160T)|>√,  var(acc180T)|>√, var(acc200T)|>√]
errorlowY = errorhighT;


ydata = [mean(acc20), mean(acc40), mean(acc60), mean(acc80), mean(acc100), mean(acc120), mean(acc140), mean(acc160), mean(acc180), mean(acc200) accOG]
ydataT = [mean(acc20T), mean(acc40T), mean(acc60T), mean(acc80T), mean(acc100T), mean(acc120T), mean(acc140T), mean(acc160T), mean(acc180T), mean(acc200T) accOGT]


fig = Figure()
ax=Axis(fig[1,1])
sca = scatter!(fig[1,1],1:length(ydata), ydata, color = :blue) 
scat =scatter!(fig[1,1],1:length(ydataT), ydataT, color = :black) 

errorbars!(1:length(errorhigh), ydata[1:7], errorhigh, errorlow,
    color = :black,
    whiskerwidth = 10)


errorbars!(1:length(errorhighT), ydataT[1:7], errorhigh, errorlow,
    color = :black,
    whiskerwidth = 10)

processors = ["20", "40", "60", "80", "100", "120", "140", "160", "180", "200" "No clustering"]
ax.xticks = (1:length(ydata), processors)
Label(fig[1,0], "Accuracy for KNN",rotation = π/2, tellheight = false)
#hidedecorations!(ax)
ax.yticks = LinearTicks(20)



Legend(fig[1, 3],
    [sca, scat],
    ["Test", "Train"])


fig