using GLMakie
using Clustering
using Distances
using Statistics

struct Picture{T<:Real}
    ID::Int
    class::Int
    data::AbstractVector{T}
    function Picture(x::AbstractVector{<:Real})
        try
            Int64(sqrt(length(x[begin+2:end])))
        catch e
            error("The input vector, with the first two values removed, can not be squared.")
        end
        return new{eltype(x)}(x[1], x[2], x[3:end])
    end
    Picture(a::Int, b::Int, c::Vector{<:Real}) = new{eltype(c)}(a, b, c)
end

import Base: getproperty
getproperty(x::Vector{<:Picture}, f::Symbol) = getproperty.(x, f)

using Random: shuffle
"""
    TrainTestSplit(pics::AbstractVector{Picture}, train_to_test_ratio::Rational, shuffle_pics::Bool=true)
    TrainTestSplit(ratio::Rational, n::Int, train::AbstractVector{Picture}, test::AbstractVector{Picture{T}})
"""
struct TrainTestSplit{T<:Real}
    ratio::Rational
    n::Int
    train::AbstractVector{Picture{T}}
    test::AbstractVector{Picture{T}}
    TrainTestSplit(ratio::Rational, n::Int, train::AbstractVector{<:Picture}, test::AbstractVector{<:Picture}) = new{T}(ratio, n, train, test)
    TrainTestSplit(train::AbstractVector{<:Picture}, test::AbstractVector{<:Picture}) = new{T}(length(train)//length(test), length(train)+length(test), train, test)
    function TrainTestSplit(pics::AbstractVector{Picture{T}}, train_to_test_ratio::Rational, shuffle_pics::Bool=true) where {T<:Real}
        if shuffle_pics
            pics = pics[shuffle(eachindex(pics))]
        end
        n_parts = train_to_test_ratio.num + train_to_test_ratio.den

        trainpics = pics[begin : train_to_test_ratio.num//n_parts * end ÷1]
        testpics = pics[(end - train_to_test_ratio.den//n_parts * end ÷1 +1) : end]
        return new{T}(train_to_test_ratio, length(pics), trainpics, testpics)
    end
end
import Base.length
length(tts::TrainTestSplit) = tts.n

testclasses(tts::TrainTestSplit) = getfield.(tts.test, :class)
trainclasses(tts::TrainTestSplit) = getfield.(tts.train, :class)

import Base: show
show(io::IO, p::Picture) = println(io, "A $(p.class) drawn by $(p.ID)")
show(io::IO, tts::TrainTestSplit) = println(io, "A TrainTestSplit object with $(tts.n) entries, of train to test ratio $(tts.ratio)")

using DataFrames
const Dflatten = DataFrames.flatten   #? To handle naming conflict with my own function
flatten(m::AbstractMatrix) = vcat(eachcol(m)...)
function unflatten(x::AbstractArray)
    try
        sidelength = sqrt(length(x)) |> Int64
        batches = [x[sidelength*(i-1)+1:sidelength*i] for i in 1:sidelength]
        return hcat(batches...)
    catch e
        error("The sidelength (`x|>length|>sqrt`) could not be converted to an integer. ")
    end
    
end

function visualize_picture(data::AbstractMatrix, colormap::Symbol=:viridis)
    fig, ax = heatmap(data|>reverse|>x->reverse(x, dims=1), figure=(resolution=(400, 400),); colormap)
    hidedecorations!(ax)
    fig
end
visualize_picture(data::AbstractVector, colormap::Symbol=:viridis) = visualize_picture(unflatten(data), colormap)
function visualize_picture(p::Picture, colormap::Symbol=:viridis)
    fig, ax = heatmap(unflatten(p.data)|>reverse|>x->reverse(x, dims=1), figure=(resolution=(400, 400),),
        axis=(title="Student ID: $(p.ID|>Int64)\nGround truth: $(p.class|>Int64)", ); colormap
    )
    hidedecorations!(ax)
    fig
end

"""
    datamat(pics::Vector{Picture}) = hcat(getfield.(pics, :data)...)

Return the data of all pictures in `pics`.
A column in the returned Matrix represents a single picture.
"""
datamat(pics::Vector{<:Picture}) = hcat(getfield.(pics, :data)...)
datamat(tts::TrainTestSplit)= (train=datamat(tts.train), test=datamat(tts.test))


"""
    classify(neighbor_inds::Vector{Int}, train_classes::Vector{Int}; tiebreaker=rand, l::Int=1, possible_classes=unique(train_classes))
    classify(neighbor_inds::Vector{Vector{Int}}, args...; kwargs...)

kwargs:
`tiebreaker` is 
1) a function that takes the tuple of candidates and returns a value, or 
2) a value that is returned upon a tie.
"""
function classify(neighbor_inds::AbstractVector{Int}, train_classes::AbstractVector{Int}; tiebreaker=rand, l::Int=1)
    possible_classes = unique(train_classes)
    neighbor_classes = train_classes[neighbor_inds]
    my_counts = [count(==(psbl_cls), neighbor_classes) for psbl_cls in possible_classes]
    A = [possible_classes my_counts]
    sorted_counts = sortslices(A, dims=1, by=x->x[2], rev=true)
    if sorted_counts[begin, end] < l
        return missing
    elseif sorted_counts[1, 2] == sorted_counts[2, 2]
        inds = [sorted_counts[i, 2] == sorted_counts[1, 2] for i in 1:size(sorted_counts, 1)]
        candidates_of_equal_count = sorted_counts[inds, :][:, 1]
        if tiebreaker isa Function
            return candidates_of_equal_count |> tiebreaker
        else
            f = (args...)->tiebreaker
            return candidates_of_equal_count |> f
        end
    else
        return sorted_counts[1, 1]
    end
end
function classify(neighbor_inds::Vector{Vector{Int}}, train_classes::AbstractVector{Int}; kwargs...)
    [classify(neighbor_inds[i], train_classes; kwargs...) for i in eachindex(neighbor_inds)]
end

using NearestNeighbors
import NearestNeighbors: knn
function knn(train_pics::Vector{<:Picture}, test_pics::Vector{<:Picture}; k::Int, tree=BruteTree, metric=Euclidean())
    mytree = tree(hcat(getfield.(train_pics, :data)...), metric)
    return knn(mytree, hcat(getfield.(test_pics, :data)...), k, true)
end
knn(tts::TrainTestSplit{<:Real}; kwargs...) = knn(tts.train, tts.test; kwargs...)


"""  #? Defining `batch` here is required for knn_threaded below
    batch(v::AbstractVector, n_batches::Int, shuffle_pics=true)

Create a vector of `n_batches` vectors, containing all of `v`.
I think that can be done better
"""
function batch(v::AbstractVector, n_batches::Int, shuffle_input=true; check_even=true, return_indices=false)
    shuffle_input  &&  (v = shuffle(v))
    check_even  &&  length(v) % n_batches != 0   &&   @warn "Number of elements not divisible by number of batches. Batches will be uneven. Set `check_even` to false to silence this warning"
    divs, rems = divrem(length(v), n_batches)
    batchlengths = fill(divs, n_batches)
    batchlengths[end-rems+1:end] .+= 1
    
    cumsums = pushfirst!(cumsum(batchlengths), 0)
    ranges = [cumsums[i]+1:cumsums[i+1] for i in 1:n_batches]
    if return_indices
        return (ranges, getindex.([v], ranges))
    else
        return getindex.([v], ranges)
    end
end


"""
    knn_threaded(train_pics::Vector{<:Picture}, test_pics::Vector{<:Picture}; k::Int, tree=BruteTree, metric=Euclidean())


"""
function knn_threaded(train_pics::Vector{<:Picture}, test_pics::Vector{<:Picture}; k::Int, tree=BruteTree, metric=Euclidean())
    mytree = tree(hcat(getfield.(train_pics, :data)...), metric)
    output = Vector{Vector{Int}}(undef, length(test_pics))
    inds, batches = batch(test_pics, 2*Threads.nthreads(), false, check_even=false, return_indices=true)
    Threads.@threads for i in 1:length(batches)
        output[inds[i]] = knn(mytree, batches[i]|>datamat, k)[1]
    end
    return output
end
knn_threaded(tts::TrainTestSplit; kwargs...) = knn_threaded(tts.train, tts.test; kwargs...)

function classify(tts::TrainTestSplit; k, l=1, tree=BruteTree, tiebreaker=rand, metric=Euclidean())
    inds = knn_threaded(tts; k, tree, metric)
    preds = classify(inds, trainclasses(tts); l, tiebreaker)
    return preds
end

#=
"""
    knn_threaded(train_pics::Vector{<:Picture}, test_pics::Vector{<:Picture}; k::Int, tree=BruteTree, metric=Euclidean())


"""
function knn_threaded(train_pics::Vector{<:Picture}, test_pics::Vector{<:Picture}; k::Int, tree=BruteTree, metric=Euclidean())
    mytree = tree(hcat(getfield.(train_pics, :data)...), metric)
    output = Vector{Vector{Int}}(undef, length(test_pics))
    Threads.@threads for i in eachindex(output)
        output[i] = knn(mytree, test_pics.data[i], k)[1]
    end
    return output
end
=#


"""
    knn_acc(tts::TrainTestSplit{<:Real}; k::Int, l::Int=1, tree=BruteTree, tiebreaker=rand, metric=Euclidean())
"""
knn_acc(tts::TrainTestSplit{<:Real}; kwargs...) = mean(classify(tts; kwargs...) .== testclasses(tts))
knn_acc(train::Vector{<:Picture}, test::Vector{<:Picture}; kwargs...) = knn_acc(TrainTestSplit(train, test), kwargs...)

"""
	knn_acc_crossvalidate(pics::Vector{Picture}, ratio::Rational = 9//1)

Shuffle `pics` ratio.num + ratio.den times, make a TrainTestSplit for each, and calculate
the mean and standard deviation of the classification accuracy of each.
"""
function knn_acc_crossvalidate(pics::Vector{<:Picture}, ratio::Rational = 9//1)
	n_runs = ratio.num + ratio.den
	temp_TTSs = [TrainTestSplit(pics, ratio) for _ in 1:n_runs]
    x = [knn_acc(tts, k=3) for tts in temp_TTSs]
    (mean=mean(x), std=std(x), n_runs = n_runs)
end

using MultivariateStats
using StatsBase

"""
    remove_constant(m::AbstractMatrix)
    remove_constant(pics::Vector{<:Picture})

Return a matrix or vector of pictures with pixel 
values which are the same for all pictures removed.
"""
function remove_constant(m::AbstractMatrix)
    bad_row_inds = Int64[]
    all_row_inds = 1:size(m, 1) |> Vector
    for i in all_row_inds
        if maximum(m[i, :]) == minimum(m[i, :])
            push!(bad_row_inds, i)
        end
    end
    reduced_data = m[deleteat!(all_row_inds, bad_row_inds), :]
    return reduced_data
end
function remove_constant(pics::Vector{<:Picture})
    datatogether = datamat(pics)
    reduced_datamat = remove_constant(datatogether)
    return [Picture(pics[i].ID, pics[i].class, reduced_datamat[:, i]) for i in eachindex(pics)]
end

"""
    normalize(p::Picture) = Picture(p.ID, p.class, (p.data .- mean(p.data))/std(p.data))
    normalize(pics::Vector{Picture})

Z-score normalize the data of a picture `p`.
"""
normalize(p::Picture) = Picture(p.ID, p.class, (p.data .- mean(p.data))/std(p.data))
function normalize(pics::Vector{<:Picture})
    alldata = datamat(pics) 
    μ, σ = alldata|> flatten|>mean, alldata|> flatten|>std
    normed_data = (alldata .- μ) ./σ
    return [Picture(pics[i].ID, pics[i].class, normed_data[:, i]) for i in eachindex(pics)]
end

using ImageFiltering
gaussian_filter(p::Picture, σ) = Picture(p.ID, p.class, imfilter(p.data|>unflatten, Kernel.gaussian(σ))|>flatten)


using Distances
#* Allow calculation of distance between two pictures:
for M in (Distances.metrics..., Distances.weightedmetrics...)
    @eval @inline (dist::$M)(a::Picture, b::Picture) = Distances._evaluate(dist, a.data, b.data, Distances.parameters(dist))
end

"""
    map_labels(ordered_labels::Vector{<:Integer}, cluster::Hclust)

Compute the labels of a dendrogram.
`ordered_labels` is the ordered classes of the objects that have been clustered.
`cluster` is the cluster that has been made by `hclust()`.

# Example:
plt2 = StatsPlots.plot(h_cluster_322)
ordered_labels = repeat(0:9, 5) |> sort
tick_labels = map_labels(ordered_labels, h_cluster_322)

StatsPlots.xticks!(h_cluster_322.order|>eachindex, tick_labels)
plt2
"""
function map_labels(ordered_labels::Vector{<:Integer}, cluster::Hclust)
    output = Vector{String}(undef, length(ordered_labels))
    for i in eachindex(output)
        generated_label = cluster.order[i]
        real_label = ordered_labels[generated_label]
        output[i] = real_label |> string
    end
    return output
end

# import Base: show
# using PrettyTables, EvalMetrics
# function show(io::IO, cm::ConfusionMatrix)
#     pretty_table(io,
#     linebreaks=true,
#     alignment=:C,
#     body_hlines = [1, 2, 3],
#     noheader=true,
#         [
#             " "                    "Actual\npositives" "Actual\nnegatives"
#             "Prediced\npositives"        cm.tp          cm.fp
#             "Prediced\nnegatives"        cm.fn          cm.tn
#             "Total"                      cm.p           cm.n
#         ]
#     )
# end

@info "utils.jl included"