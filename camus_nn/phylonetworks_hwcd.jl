# Exact PhyloNetworks HWCD worker. The Python client owns process lifetime/logs.
using PhyloNetworks
using SHA

pkgversion(PhyloNetworks) == v"1.4.0" || error("HWCD requires PhyloNetworks 1.4.0")
source = joinpath(dirname(pathof(PhyloNetworks)), "compareNetworks.jl")
println("READY\t", pkgversion(PhyloNetworks), "\t", VERSION, "\t",
        bytes2hex(sha256(read(source))))
flush(stdout)

for line in eachline(stdin)
    fields = split(line, '\t'; keepempty=true)
    id = first(fields)
    try
        length(fields) == 4 || error("Invalid HWCD request")
        fields[2] in ("true", "false") || error("Invalid rooted argument")
        rooted = fields[2] == "true"
        # Newick readers may print diagnostics; keep stdout exclusively protocol.
        distance, true_k, inferred_k = redirect_stdout(stderr) do
            truth = readnewick(String(hex2bytes(fields[3])))
            estimate = readnewick(String(hex2bytes(fields[4])))
            Set(tiplabels(truth)) == Set(tiplabels(estimate)) ||
                error("Inferred and true network taxa differ")
            value = hardwiredclusterdistance(truth, estimate, rooted)
            (value, truth.numhybrids, estimate.numhybrids)
        end
        println("RESULT\t", id, "\tok\t", distance, "\t", true_k, "\t", inferred_k)
    catch exception
        exception isa InterruptException && rethrow()
        message = sprint(showerror, exception)
        println("RESULT\t", id, "\terror\t", bytes2hex(codeunits(message)))
    end
    flush(stdout)
end
