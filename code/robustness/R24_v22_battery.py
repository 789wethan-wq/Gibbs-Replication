"""R24 — V22 battery (4 tests).
  T1: joint crisis exclusion (drop 2000-2002 & 2008-2009) — T·ΔS Wald, accounting & price ΔH.
  T2: effective high-T EPISODE count (contiguous T>p75 runs, not months).
  T3: L/S transaction-cost estimate — high-iVol quintile turnover, net @ 20bps round-trip.
  T4: frequency-matched asymmetric test — β_ΔS~T quarterly in the S&P-500 subset.
Outputs: results/revision/R24_v22_battery.txt
"""
import os, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import chi2
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA="../data"; OUT="../results/revision"; os.makedirs(OUT,exist_ok=True)
LOG=[]
def say(*a): s=" ".join(str(x) for x in a); print(s); LOG.append(s)
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
    G=len(np.unique(g)); sc=(G/(G-1))*((len(X)-1)/(len(X)-X.shape[1])); return sc*xi@B@xi
def wald_txds(sub, dh, ds, tx, y, dc):
    s=sub.dropna(subset=[dh,ds,tx,y])
    if len(s)<200: return None
    X=np.column_stack([np.ones(len(s)),s[dh],s[ds],s[tx]]); yy=s[y].values
    b,*_=np.linalg.lstsq(X,yy,rcond=None); r=yy-X@b
    V=clu(X,r,pd.Categorical(s[dc].astype(str)).codes)
    t=b[3]/np.sqrt(V[3,3]); return b[3], t, 1-chi2.cdf(t**2,1), len(s), s[dc].nunique()
def nw_t(series,lags):
    s=pd.Series(series).dropna(); n=len(s); mn=s.mean(); v=(s**2).mean()-mn**2
    for l in range(1,min(lags+1,n)): v+=2*(1-l/(lags+1))*((s.iloc[l:].values-mn)*(s.iloc[:-l].values-mn)).mean()
    return mn, mn/np.sqrt(max(v,1e-30)/n), n
def bic_lag(resid,maxlag=8):
    r=pd.Series(resid).dropna(); n=len(r); bp,bb=0,np.inf
    for p in range(0,min(maxlag,n//4)+1):
        try:
            if p==0: rss=float(((r-r.mean())**2).sum()); k=1
            else: mdl=sm.tsa.AutoReg(r,lags=p,old_names=False).fit(); rss=float((mdl.resid**2).sum()); k=p+1
            bic=n*np.log(rss/n)+k*np.log(n)
            if bic<bb: bb,bp=bic,p
        except: pass
    return bp

m=pd.read_parquet(f"{DATA}/merged_with_accounting.parquet"); m["date"]=pd.to_datetime(m["date"])
m["dH_gpm_z"]=cs_wz(m,"dH_gpm","date")
if "TxDS" not in m.columns: m["TxDS"]=m["T"]*m["DS_z"]
say("="*68); say("R24 — V22 BATTERY"); say("="*68)

# ═══ TEST 1 — joint crisis exclusion ════════════════════════════════════════
say("\n=== TEST 1: Joint Crisis Exclusion (drop 2000-2002 & 2008-2009) ===")
crisis_yrs={2000,2001,2002,2008,2009}
keep=m[~m["date"].dt.year.isin(crisis_yrs)].copy()
say(f"  Months: full={m['date'].nunique()}  ex-crisis={keep['date'].nunique()}  "
    f"(dropped {m['date'].nunique()-keep['date'].nunique()})")
for lab,dh in [("ACCOUNTING ΔH (dH_gpm_z)","dH_gpm_z"),("PRICE-based ΔH (DH_z)","DH_z")]:
    full=wald_txds(m,dh,"DS_z","TxDS","ret_next_month","date")
    exc =wald_txds(keep,dh,"DS_z","TxDS","ret_next_month","date")
    say(f"  {lab}:")
    say(f"    full sample:  β(T·ΔS)={full[0]:+.4f}  t={full[1]:+.2f}  Wald p={full[2]:.4f}  (N={full[3]:,}, T={full[4]})")
    say(f"    ex-crisis:    β(T·ΔS)={exc[0]:+.4f}  t={exc[1]:+.2f}  Wald p={exc[2]:.4f}  (N={exc[3]:,}, T={exc[4]})")

# ═══ TEST 2 — effective high-T episode count ════════════════════════════════
say("\n=== TEST 2: Effective High-T Episode Count (T>p75 contiguous runs) ===")
T=m.groupby("date")["T"].first().sort_index()
thr=T.quantile(0.75)
above=(T>=thr).astype(int)
say(f"  Threshold T(p75)={thr:.4f}; months above={above.sum()} of {len(T)}")
def episodes(mask_series, gap=0):
    idx=list(mask_series.index); vals=mask_series.values; eps=[]; cur=None; last_hi=None
    for i,(d,v) in enumerate(zip(idx,vals)):
        if v==1:
            if cur is None: cur=[d,d]
            else: cur[1]=d
            last_hi=i
        else:
            if cur is not None and gap>0 and last_hi is not None and i-last_hi<=gap:
                continue
            if cur is not None: eps.append(tuple(cur)); cur=None
    if cur is not None: eps.append(tuple(cur))
    return eps
for gap,lab in [(0,"strict contiguous"),(1,"merge 1-month gaps")]:
    eps=episodes(above,gap)
    say(f"  [{lab}] {len(eps)} independent high-T episodes:")
    for s,e in eps:
        sub=T[(T.index>=s)&(T.index<=e)]
        say(f"     {s.strftime('%Y-%m')}..{e.strftime('%Y-%m')}  ({len(sub)} mo, peak T={sub.max():.3f})")
say(f"  -> The Wald is effectively identified by ~{len(episodes(above,1))} temperature events,")
say(f"     not {above.sum()} independent months (months within an episode are highly serially dep.).")

# ═══ TEST 3 — transaction-cost estimate ═════════════════════════════════════
say("\n=== TEST 3: Transaction Cost — high-iVol quintile turnover, net @ 20bps RT ===")
COST=0.0020  # 20 bps round-trip
def quintile_turnover_ls(panel, sortcol, dc="date", ret="ret_next_month"):
    d=panel.dropna(subset=[sortcol,ret]).copy()
    d["qd"]=d.groupby(dc)[sortcol].transform(lambda x: pd.qcut(x,5,labels=False,duplicates="drop") if x.nunique()>=5 else np.nan)
    d=d.dropna(subset=["qd"])
    dates=sorted(d[dc].unique())
    q5_prev=q1_prev=None; to5=[]; to1=[]; ls=[]
    for dt in dates:
        g=d[d[dc]==dt]
        q5=set(g[g.qd==4]["stock_id" if "stock_id" in g else "ticker"]); q1=set(g[g.qd==0]["stock_id" if "stock_id" in g else "ticker"])
        ls.append(g[g.qd==4][ret].mean()-g[g.qd==0][ret].mean())
        if q5_prev is not None and len(q5)>0: to5.append(len(q5-q5_prev)/len(q5))
        if q1_prev is not None and len(q1)>0: to1.append(len(q1-q1_prev)/len(q1))
        q5_prev,q1_prev=q5,q1
    return np.mean(to5), np.mean(to1), np.array(ls)
t5,t1,ls=quintile_turnover_ls(m,"DS_z")
gross_m=np.nanmean(ls); cost_m=(t5+t1)*COST
say(f"  S&P monthly, sort on DS_z (iVol):")
say(f"    avg one-way turnover: high-iVol Q5={t5*100:.1f}%/mo, low-iVol Q1={t1*100:.1f}%/mo")
say(f"    gross L/S (Q5-Q1): {gross_m*100:+.3f}%/mo ({gross_m*1200:+.1f}%/yr)")
say(f"    monthly cost @ 20bps RT (both legs): {cost_m*100:.3f}%/mo ({cost_m*1200:.1f}%/yr)")
say(f"    NET L/S: {(gross_m-cost_m)*100:+.3f}%/mo ({(gross_m-cost_m)*1200:+.1f}%/yr)")
# survivorship-free quarterly OOS net (gross +4.1%/yr from R21)
q=pd.read_parquet(f"{DATA}/merged_sf1_quarterly_survfree.parquet")
tq5,tq1,lsq=quintile_turnover_ls(q,"delta_s_z",dc="q",ret="ret_next")
say(f"  Survivorship-free quarterly, sort on ΔS:")
say(f"    avg one-way turnover/quarter: Q5={tq5*100:.1f}%, Q1={tq1*100:.1f}%")
say(f"    gross L/S: {np.nanmean(lsq)*100:+.3f}%/q ({np.nanmean(lsq)*400:+.1f}%/yr)")
costq=(tq5+tq1)*COST
say(f"    NET L/S: {(np.nanmean(lsq)-costq)*100:+.3f}%/q ({(np.nanmean(lsq)-costq)*400:+.1f}%/yr)")
say(f"  NOTE: 20bps RT is optimistic for high-iVol small caps; real costs likely higher.")

# ═══ TEST 4 — frequency-matched asymmetric test (S&P quarterly) ═════════════
say("\n=== TEST 4: Frequency-matched β_ΔS~T (S&P-500 subset, quarterly) ===")
sp_tickers=set(m["stock_id"].unique())
qsp=q[q["ticker"].isin(sp_tickers)].dropna(subset=["ret_next","delta_h_z","delta_s_z","T"]).copy()
say(f"  S&P subset of R18 quarterly panel: {qsp['ticker'].nunique()} tickers, "
    f"{qsp['q'].nunique()} quarters, N={len(qsp):,}")
# step 1: quarterly FM betas
rec=[]
for d,g in qsp.groupby("q"):
    s=g[["ret_next","delta_h_z","delta_s_z","T"]].dropna()
    if len(s)<20: continue
    r=sm.OLS(s["ret_next"],sm.add_constant(s[["delta_h_z","delta_s_z"]])).fit()
    rec.append((d,r.params["delta_h_z"],r.params["delta_s_z"],s["T"].iloc[0]))
b=pd.DataFrame(rec,columns=["q","b_dH","b_dS","T"])
# step 2: HAC (BIC lag)
for lab,col in [("β_ΔH ~ T","b_dH"),("β_ΔS ~ T","b_dS")]:
    y=b[col].values; X=sm.add_constant(b[["T"]].values)
    ols=sm.OLS(y,X).fit(); p=max(bic_lag(ols.resid),1)
    hac=sm.OLS(y,X).fit(cov_type="HAC",cov_kwds={"maxlags":p})
    say(f"  {lab}: slope={ols.params[1]:+.4f} R²={ols.rsquared:.3f} OLS t={ols.tvalues[1]:+.2f} "
        f"HAC t={hac.tvalues[1]:+.2f} (lag={p}) n={int(ols.nobs)}")
say(f"  COMPARISON (β_ΔS~T HAC t):")
say(f"    S&P monthly       : +2.25   (R20)")
say(f"    S&P quarterly     : (this test, above)")
say(f"    Full-univ quarterly: +2.70  (R20/R18)")
say(f"  -> Isolates frequency/construction vs universe in the asymmetric test.")

# ═══ BOTTOM LINE ════════════════════════════════════════════════════════════
say("\n=== BOTTOM LINE ===")
say("Test 1: crisis-excluded Wald p-values reported (accounting & price).")
say(f"Test 2: ~{len(episodes(above,1))} independent high-T episodes identify the Wald (not {above.sum()} months).")
say(f"Test 3: high-iVol L/S net of 20bps — S&P net {(gross_m-cost_m)*1200:+.1f}%/yr, surv-free net {(np.nanmean(lsq)-costq)*400:+.1f}%/yr.")
say("Test 4: frequency-matched β_ΔS~T (S&P quarterly) reported above.")
with open(f"{OUT}/R24_v22_battery.txt","w") as f: f.write("\n".join(LOG)+"\n")
say(f"\nSaved: {OUT}/R24_v22_battery.txt")
