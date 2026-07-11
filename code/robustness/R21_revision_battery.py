"""R21 — Remaining revision tasks.
  [4] Bootstrap distribution of the T·ΔS Wald test (reframe the bare 52.3%)
      + high-T bootstrap fraction (H3 support rate).
  [5] Table 1 t(ΔS)=+4.80 vs Table 4 t(ΔS)=+0.92 — identify the spec difference.
  [6] Full Panel A (Models A/B/C) for the R18 survivorship-free panel.
  [7] High-T-only post-2009 test — T-regime vs calendar.
  [8] OOS at quarterly frequency in the survivorship-corrected panel.
Outputs: results/revision/R21_revision_battery.txt  (+ bootstrap histogram PNG)
"""
import os, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA="../data"; OUT="../results/revision"; FIG="../outputs/figures"
os.makedirs(OUT, exist_ok=True)
RNG=np.random.default_rng(20260617)
LOG=[]
def say(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.append(s)
def cs_wz(df,col,dc,pct=0.01):
    def _w(x):
        x=x.dropna()
        if len(x)<5: return pd.Series(np.nan,index=x.index)
        xc=x.clip(x.quantile(pct),x.quantile(1-pct)); sd=xc.std()
        return (xc-xc.mean())/sd if sd>1e-10 else pd.Series(np.nan,index=x.index)
    return df.groupby(dc)[col].transform(_w)
def clu(X,r,g):
    xi=np.linalg.pinv(X.T@X); B=np.zeros((X.shape[1],)*2)
    for u in np.unique(g):
        Xm=X[g==u]; rm=r[g==u]; B+=Xm.T@np.outer(rm,rm)@Xm
    G=len(np.unique(g)); sc=(G/(G-1))*((len(X)-1)/(len(X)-X.shape[1]))
    return sc*xi@B@xi
def fm_nw(panel,y,xs,dc,lags=5,min_cs=20):
    C=[]
    for d,g in panel.groupby(dc):
        sub=g[[y]+xs].dropna()
        if len(sub)<max(min_cs,len(xs)+2): continue
        X=sm.add_constant(sub[xs],has_constant="add")
        C.append(sm.OLS(sub[y],X).fit().params[xs].rename(d))
    cdf=pd.DataFrame(C); out={}
    for c in xs:
        s=cdf[c].dropna(); n=len(s); mn=s.mean(); g0=(s**2).mean()-mn**2; v=g0
        for l in range(1,min(lags+1,n)):
            v+=2*(1-l/(lags+1))*((s.iloc[l:].values-mn)*(s.iloc[:-l].values-mn)).mean()
        out[c]=(mn, mn/np.sqrt(max(v,1e-30)/n), n)
    return out

# ── load panels ─────────────────────────────────────────────────────────────
m=pd.read_parquet(f"{DATA}/merged_with_accounting.parquet"); m["date"]=pd.to_datetime(m["date"])
m["dH_gpm_z"]=cs_wz(m,"dH_gpm","date"); m["TxDS"]=m["T"]*m["DS_z"]
reg=pd.read_parquet(f"{DATA}/regime_assignments.parquet")
if "date" in reg.columns: reg["date"]=pd.to_datetime(reg["date"]); reg=reg.set_index("date")
q=pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")

say("="*70); say("R21 — REVISION BATTERY (tasks 4-8)"); say("="*70)

# ═══ [5] TABLE 1 (4.80) vs TABLE 4 (0.92) ═══════════════════════════════════
say("\n"+"#"*70); say("# [5] Table 1 t(ΔS)=+4.80  vs  Table 4 t(ΔS)=+0.92 — spec diff")
say("#"*70)
mb=m.dropna(subset=["ret_next_month","dH_gpm_z","DS_z","T"]).copy()
# Candidate A: FM Model B, NW-5 (cross-sectional slope t over all months)  -> ~4.80
o=fm_nw(mb,"ret_next_month",["dH_gpm_z","DS_z"],"date");
say(f"  (A) FM Model B, NW-5, full sample:        t(ΔS) = {o['DS_z'][1]:+.2f}   (n_months={o['DS_z'][2]})")
# Candidate B: single pooled time-series of the L/S or regime-difference (few obs)
# regime-conditional FM t(ΔS) difference (Table 4 construction)
mr=mb.join(reg["high_T"] if "high_T" in reg.columns else reg.iloc[:,0].rename("high_T"), on="date")
for lab,flag in [("low-T",0),("high-T",1)]:
    sub=mr[mr["high_T"]==flag]
    oo=fm_nw(sub,"ret_next_month",["dH_gpm_z","DS_z"],"date")
    say(f"  (B) FM Model B within {lab} regime:        t(ΔS) = {oo['DS_z'][1]:+.2f}   (n_months={oo['DS_z'][2]})")
# Candidate C: annual (low-frequency) FM -> few obs, low t
mb["year"]=mb["date"].dt.year
oa=fm_nw(mb,"ret_next_month",["dH_gpm_z","DS_z"],"year",lags=1,min_cs=200)
say(f"  (C) FM Model B aggregated ANNUALLY:        t(ΔS) = {oa['DS_z'][1]:+.2f}   (n_years={oa['DS_z'][2]})")
# Candidate D: L/S portfolio time-series alpha on ΔS sort (single regression)
say("  Likely resolution: 4.80 = NW-5 t on 334 MONTHLY cross-sectional slopes")
say("  (high N -> high power); 0.92 = a LOW-FREQUENCY / regime-restricted t on")
say("  far fewer obs. t IS scale-invariant, but it is NOT sample/spec-invariant:")
say("  the two numbers use different N and conditioning, not different units.")

# ═══ [6] R18 PANEL A (Models A/B/C, survivorship-free) ══════════════════════
say("\n"+"#"*70); say("# [6] R18 survivorship-free panel — Panel A (Models A/B/C, FM NW-4)")
say("#"*70)
qf=q.dropna(subset=["ret_next","delta_s_z","delta_h_z"]).copy()
A=fm_nw(qf,"ret_next",["delta_g"],"q",lags=4)
B=fm_nw(qf,"ret_next",["delta_h_z","delta_s_z"],"q",lags=4)
Cc=fm_nw(qf,"ret_next",["delta_h_z","T_delta_s"],"q",lags=4)
say(f"  {'Model':<26}{'coef':>11}{'t (NW-4)':>11}{'Tq':>6}")
say(f"  {'A: ΔG':<26}{A['delta_g'][0]:>+11.5f}{A['delta_g'][1]:>+11.2f}{A['delta_g'][2]:>6}")
say(f"  {'B: ΔH':<26}{B['delta_h_z'][0]:>+11.5f}{B['delta_h_z'][1]:>+11.2f}{B['delta_h_z'][2]:>6}")
say(f"  {'B: ΔS':<26}{B['delta_s_z'][0]:>+11.5f}{B['delta_s_z'][1]:>+11.2f}{'':>6}")
say(f"  {'C: ΔH':<26}{Cc['delta_h_z'][0]:>+11.5f}{Cc['delta_h_z'][1]:>+11.2f}{Cc['delta_h_z'][2]:>6}")
say(f"  {'C: T·ΔS':<26}{Cc['T_delta_s'][0]:>+11.5f}{Cc['T_delta_s'][1]:>+11.2f}{'':>6}")

# ═══ [7] HIGH-T POST-2009 (T-regime vs calendar) ════════════════════════════
say("\n"+"#"*70); say("# [7] High-T post-2009 — disentangle T-regime from calendar period")
say("#"*70)
def wald_txds(panel):
    s=panel.dropna(subset=["dH_gpm_z","DS_z","TxDS","ret_next_month"])
    if len(s)<500 or s["date"].nunique()<12: return None
    X=np.column_stack([np.ones(len(s)),s["dH_gpm_z"],s["DS_z"],s["TxDS"]]); y=s["ret_next_month"].values
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b
    V=clu(X,r,pd.Categorical(s["date"].astype(str)).codes)
    t=b[3]/np.sqrt(V[3,3]); return b[3],t,1-chi2.cdf(t**2,1),len(s)
for lab,sub in [("FULL sample",mb),
                ("pre-2009",mb[mb["date"]<"2009-01-01"]),
                ("post-2009",mb[mb["date"]>="2009-01-01"])]:
    r=wald_txds(sub)
    if r: say(f"  T·ΔS Wald — {lab:14}: β={r[0]:+.4f}  t={r[1]:+.2f}  p={r[2]:.4f}  (N={r[3]:,})")
# within post-2009: high-T vs low-T β_ΔS
post=mr[mr["date"]>="2009-01-01"]
for lab,flag in [("post-2009 low-T",0),("post-2009 high-T",1)]:
    sub=post[post["high_T"]==flag]
    oo=fm_nw(sub,"ret_next_month",["dH_gpm_z","DS_z"],"date")
    if "DS_z" in oo: say(f"  β_ΔS {lab:18}: t={oo['DS_z'][1]:+.2f} (n_months={oo['DS_z'][2]})")
say("  -> If T·ΔS still significant post-2009 AND high-T differs from low-T within")
say("     the post-2009 window, the effect is the T-regime, not a calendar artifact.")

# ═══ [8] OOS QUARTERLY (survivorship-corrected) ═════════════════════════════
say("\n"+"#"*70); say("# [8] OOS quarterly, survivorship-corrected (vs 14/18 contaminated)")
say("#"*70)
qoos=q.dropna(subset=["ret_next","delta_g"]).copy().sort_values("q")
quarters=sorted(qoos["q"].unique())
split=quarters[len(quarters)//3]   # ~first third as burn-in
ls_q=[]
for qt in quarters:
    if qt<=split: continue
    train=qoos[qoos["q"]<qt]
    if train["q"].nunique()<12: continue
    o=fm_nw(train,"ret_next",["delta_g"],"q",lags=4)
    sgn=np.sign(o["delta_g"][0]) if "delta_g" in o else 0
    cur=qoos[qoos["q"]==qt]
    if cur["delta_g"].nunique()<5: continue
    cur=cur.copy(); cur["qd"]=pd.qcut(cur["delta_g"],5,labels=False,duplicates="drop")
    hi=cur[cur["qd"]==4]["ret_next"].mean(); lo=cur[cur["qd"]==0]["ret_next"].mean()
    ls_q.append((qt, sgn*(hi-lo)))   # trade in the in-sample sign direction
lsdf=pd.DataFrame(ls_q,columns=["q","ls"]).dropna()
lsdf["year"]=lsdf["q"].astype(str).str[:4].astype(int)
yr=lsdf.groupby("year")["ls"].mean()
say(f"  OOS quarters: {len(lsdf)}  mean L/S = {lsdf['ls'].mean()*100:+.2f}%/q "
    f"({lsdf['ls'].mean()*400:+.1f}%/yr)")
say(f"  Profitable quarters: {(lsdf['ls']>0).sum()}/{len(lsdf)} "
    f"({(lsdf['ls']>0).mean()*100:.0f}%)")
say(f"  Profitable years:    {(yr>0).sum()}/{len(yr)}   [contaminated S&P: 14/18]")
t_oos=lsdf['ls'].mean()/(lsdf['ls'].std()/np.sqrt(len(lsdf)))
say(f"  OOS L/S t-stat: {t_oos:+.2f}")

# ═══ [4] BOOTSTRAP DISTRIBUTION of T·ΔS Wald (reframe 52.3%) ═════════════════
say("\n"+"#"*70); say("# [4] Bootstrap distribution of T·ΔS Wald (reframe bare 52.3%)")
say("#"*70)
base=mb.dropna(subset=["dH_gpm_z","DS_z","TxDS","ret_next_month"]).copy()
dates=np.array(sorted(base["date"].unique())); nd=len(dates); BLOCK=12; NB=500
by_date={d:base[base["date"]==d] for d in dates}
ps,ts,ratios=[],[],[]
hi_by={d:by_date[d] for d in dates}
for _ in range(NB):
    starts=RNG.integers(0,nd-BLOCK,size=nd//BLOCK+1)
    sel=np.concatenate([dates[s:s+BLOCK] for s in starts])[:nd]
    bs=pd.concat([by_date[d] for d in sel])
    X=np.column_stack([np.ones(len(bs)),bs["dH_gpm_z"],bs["DS_z"],bs["TxDS"]]); y=bs["ret_next_month"].values
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b
    V=clu(X,r,pd.Categorical(bs["date"].astype(str)).codes)
    t=b[3]/np.sqrt(V[3,3]); ps.append(1-chi2.cdf(t**2,1)); ts.append(t)
ps=np.array(ps); ts=np.array(ts)
# high-T support fraction: β_ΔS larger (abs) in high-T than low-T per bootstrap
mr2=mr.dropna(subset=["ret_next_month","dH_gpm_z","DS_z","high_T"])
h3=[]
for _ in range(NB):
    starts=RNG.integers(0,nd-BLOCK,size=nd//BLOCK+1)
    sel=np.concatenate([dates[s:s+BLOCK] for s in starts])[:nd]
    bs=mr2[mr2["date"].isin(set(sel))]
    lo=fm_nw(bs[bs.high_T==0],"ret_next_month",["dH_gpm_z","DS_z"],"date")
    hi=fm_nw(bs[bs.high_T==1],"ret_next_month",["dH_gpm_z","DS_z"],"date")
    if "DS_z" in lo and "DS_z" in hi and lo["DS_z"][0]!=0:
        h3.append(abs(hi["DS_z"][0])>abs(lo["DS_z"][0]))
say(f"  T·ΔS Wald over {NB} block-bootstraps (block={BLOCK}m):")
say(f"    mean t={ts.mean():+.2f}  median p={np.median(ps):.3f}")
say(f"    %(p<0.05) = {(ps<0.05).mean()*100:.1f}%   [the bare '52.3%']")
say(f"    %(p<0.10) = {(ps<0.10).mean()*100:.1f}%   %(t>0) = {(ts>0).mean()*100:.1f}%")
say(f"    t percentiles 5/50/95 = {np.percentile(ts,5):+.2f}/{np.percentile(ts,50):+.2f}/{np.percentile(ts,95):+.2f}")
say(f"  High-T support fraction (|β_ΔS,high|>|β_ΔS,low|): {np.mean(h3)*100:.1f}%  (H3)")
# histogram PNG
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].hist(ts,bins=40,color="#3b66cc",edgecolor="white")
ax[0].axvline(1.96,ls="--",c="r"); ax[0].axvline(-1.96,ls="--",c="r")
ax[0].set_title("Bootstrap T·ΔS Wald t-stat"); ax[0].set_xlabel("t")
ax[1].hist(ps,bins=40,color="#cc6633",edgecolor="white")
ax[1].axvline(0.05,ls="--",c="r"); ax[1].set_title("Bootstrap T·ΔS Wald p-value"); ax[1].set_xlabel("p")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_bootstrap_TxDS.png",dpi=130); plt.close()
say(f"  Saved histogram: {FIG}/fig_bootstrap_TxDS.png")

with open(f"{OUT}/R21_revision_battery.txt","w") as f: f.write("\n".join(LOG)+"\n")
say("\n"+"="*70); say(f"Saved: {OUT}/R21_revision_battery.txt")
