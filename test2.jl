import SymmetryCode as S
using JLD2

case = S.case_snellius()

dns  = S.dns_runs().train[1]          # (; visc=1.5e-4, seed=1, role=:train)

S.create_dns(case, dns)               # warm-up → dnsfile
S.create_data(case, dns)              # one pass → fields/meta per Δ ∈ {2,3,4}

load(S.fieldsfile(case, dns, 3.0), "redelta")'   # per-snapshot global Re_Δ at Δ/h=3
