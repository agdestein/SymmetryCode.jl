import SymmetryCode as S

case = S.case_snellius()

# one model, baseline (no Re_Δ):
m = (; arch=:conv, tier=:saturated, netseed=0, use_redelta=false)
# S.train_model(case, m, S.build_trainpool(case))           # → psfile(case, m)
S.train_model(case, m, [(S.dns_runs().train[1], 3.0)])           # → psfile(case, m)
load(S.psfile(case, m), "losses_valid")                   # convergence curve
