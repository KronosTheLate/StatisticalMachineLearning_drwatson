"""
    print_fields(x, l_max=500, padding=15, padfunc=lpad)
Print all fieldnames of `x`, followed their value.
"""
function print_fields(x, l_max=500, padding=15, padfunc=lpad)
    fields = fieldnames(x)
    for field in fields
        second_part = "$(getfield(x, field))"
        second_part = length(second_part) < l_max ? second_part : "More than $l_max letters."
        println(padfunc("$field = ", padding) *  second_part)
    end
end

"""
    ⊕(args...) = 1/sum(x->1/x, args)
Compute the reciprocal over sum of reciprocal.
Can be used as infix operator.
"""
⊕(args...) = 1/sum(x->1/x, args)
export ⊕

"""
    tofn(x) = typeof(fieldnames(x))
"""
tofn(x) = typeof(fieldnames(x))