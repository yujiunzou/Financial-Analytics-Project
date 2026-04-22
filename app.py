import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle, os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & STYLE
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FraudSight",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #e0e4ef !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: #1a1f2e !important;
    border: 1px solid #2e3450 !important;
    color: #fff !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #3b5bdb, #4c6ef5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(135deg, #4c6ef5, #748ffc) !important;
}

.main { background: #f8f9fc; }
.block-container { padding-top: 1.5rem !important; }

.fs-card {
    background: white;
    border-radius: 12px;
    padding: 22px 26px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 18px;
}

.badge-low  { background:#d3f9d8; color:#1a7a30; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.badge-mod  { background:#fff3bf; color:#7d5a00; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }
.badge-high { background:#ffe0e0; color:#c0392b; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }

.page-title    { font-family:'DM Serif Display',serif; font-size:2rem; color:#0f1117; margin-bottom:0.2rem; }
.page-subtitle { color:#6b7280; font-size:0.95rem; margin-bottom:1.2rem; }

.warn-banner {
    background:#fff5f5; border:1px solid #fca5a5;
    border-left:4px solid #ef4444; border-radius:8px;
    padding:10px 16px; color:#7f1d1d; font-size:0.88rem; margin-bottom:12px;
}

[data-testid="stMetric"] {
    background: white; border-radius: 10px;
    padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = "fraud_model.pkl"

FEATURES = [
    'roa','profit_margin','current_ratio','debt_ratio','asset_turnover',
    'ocf_ratio','sga_ratio','depr_ratio',
    'revenue_growth','asset_growth','income_growth',
    'accrual_ratio','cfo_to_income','receivable_ratio','profit_margin_vs_industry'
]

FEATURE_LABELS = {
    'roa':                       'Return on Assets (ROA)',
    'profit_margin':             'Profit Margin',
    'current_ratio':             'Current Ratio',
    'debt_ratio':                'Debt Ratio',
    'asset_turnover':            'Asset Turnover',
    'ocf_ratio':                 'Operating Cash Flow Ratio',
    'sga_ratio':                 'SG&A Ratio',
    'depr_ratio':                'Depreciation Ratio',
    'revenue_growth':            'Revenue Growth',
    'asset_growth':              'Asset Growth',
    'income_growth':             'Income Growth',
    'accrual_ratio':             'Accrual Ratio',
    'cfo_to_income':             'CFO to Net Income',
    'receivable_ratio':          'Receivable Ratio',
    'profit_margin_vs_industry': 'Profit Margin vs Industry',
}

TREND_METRICS = {
    'Total Revenue':         'Total Revenue ($M)',
    'Net Income':            'Net Income ($M)',
    'Total Assets':          'Total Assets ($M)',
    'Operating Cash Flow':   'Operating Cash Flow ($M)',
    'ROA':                   'Return on Assets',
    'Profit Margin':         'Profit Margin',
    'Debt Ratio':            'Debt Ratio',
    'Current Ratio':         'Current Ratio',
    'Asset Turnover':        'Asset Turnover',
    'Accrual Ratio':         'Accrual Ratio',
}

METRIC_INTERP = {
    'Total Revenue':       "Revenue trend reflects top-line growth momentum. Consistent growth is positive; sudden spikes may warrant scrutiny.",
    'Net Income':          "Net income shows overall profitability. Divergence from revenue (income rising while revenue falls) can signal earnings manipulation.",
    'Total Assets':        "Asset growth should roughly align with business expansion. Unusually rapid asset inflation may indicate aggressive accounting.",
    'Operating Cash Flow': "Cash flow from operations reflects actual cash generation. If it consistently lags net income, earnings quality may be poor.",
    'ROA':                 "Return on Assets measures how efficiently the company uses assets to generate profit. Declining ROA signals deteriorating performance.",
    'Profit Margin':       "Profit margin shows how much of revenue becomes profit. Sudden improvements without clear business reasons can be a warning sign.",
    'Debt Ratio':          "Debt Ratio = Total Liabilities / Total Assets. A rising ratio means increasing financial leverage and risk.",
    'Current Ratio':       "Current Ratio = Current Assets / Current Liabilities. Values below 1.0 suggest potential liquidity issues.",
    'Asset Turnover':      "Asset Turnover = Revenue / Total Assets. Declining turnover may indicate overinvestment or stagnating sales.",
    'Accrual Ratio':       "Accrual Ratio = (Net Income – Operating CFO) / Assets. High positive values suggest earnings are driven by accruals — a classic fraud signal.",
}

IND_BENCH = {
    'Manufacturing':    dict(roa=0.07,profit_margin=0.09,current_ratio=1.6,debt_ratio=0.45,asset_turnover=0.85),
    'Technology':       dict(roa=0.10,profit_margin=0.18,current_ratio=2.2,debt_ratio=0.35,asset_turnover=0.70),
    'Finance':          dict(roa=0.02,profit_margin=0.20,current_ratio=1.1,debt_ratio=0.85,asset_turnover=0.15),
    'Services':         dict(roa=0.08,profit_margin=0.10,current_ratio=1.4,debt_ratio=0.50,asset_turnover=1.00),
    'Wholesale/Retail': dict(roa=0.06,profit_margin=0.04,current_ratio=1.5,debt_ratio=0.55,asset_turnover=1.80),
    'Transportation':   dict(roa=0.05,profit_margin=0.07,current_ratio=1.2,debt_ratio=0.60,asset_turnover=0.90),
    'Other':            dict(roa=0.06,profit_margin=0.08,current_ratio=1.5,debt_ratio=0.50,asset_turnover=0.80),
}

PALETTE = ['#3b5bdb','#10b981','#f59e0b','#ef4444','#8b5cf6',
           '#06b6d4','#ec4899','#84cc16','#f97316','#64748b']

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

def safe_get(df, key, col=0):
    try:
        val = df.loc[key].iloc[col]
        return float(val) if not pd.isna(val) else 0.0
    except Exception:
        return 0.0

def compute_beneish(r, l):
    """Original Beneish M-Score — same as working Colab version."""
    try:
        DSRI = (r['rect']/r['revt']) / (l['rect']/l['revt'])
        gm   = (r['revt']-r['cogs'])/r['revt']
        gml  = (l['revt']-l['cogs'])/l['revt']
        GMI  = gml/gm
        AQI  = (1-(r['act']+r['ppent'])/r['at']) / (1-(l['act']+l['ppent'])/l['at'])
        SGI  = r['revt']/l['revt']
        DEPI = (l['dpc']/(l['dpc']+l['ppent'])) / (r['dpc']/(r['dpc']+r['ppent']))
        SGAI = (r['xsga']/r['revt']) / (l['xsga']/l['revt'])
        LVGI = (r['lt']/r['at']) / (l['lt']/l['at'])
        TATA = (r['ni']-r['oancf'])/r['at']
        return round(-4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
                     + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI, 4), {}
    except Exception as e:
        return None, {"error": str(e)}

@st.cache_data(ttl=3600)
def fetch_company(ticker_sym: str):
    """Fetch via yfinance with proper error handling."""
    import yfinance as yf, time

    for attempt in range(3):
        try:
            t    = yf.Ticker(ticker_sym)
            info = t.fast_info   # fast_info is lighter and less rate-limited
            name = getattr(info, 'quote_type', None) and ticker_sym.upper()

            # Try to get company name from info dict
            try:
                full_info = t.info
                name = full_info.get('longName', ticker_sym.upper())
                sector   = full_info.get('sector', 'Other')
                country  = full_info.get('country', 'N/A')
                employees= full_info.get('fullTimeEmployees', 0)
            except Exception:
                name = ticker_sym.upper()
                sector, country, employees = 'Other', 'N/A', 0

            inc = t.financials
            bal = t.balance_sheet
            cf  = t.cashflow

            if inc is None or inc.empty:
                if attempt < 2:
                    time.sleep(8)
                    continue
                return None,None,None,None,None,"No financial data returned. Check the ticker symbol."
            if inc.shape[1] < 2:
                return None,None,None,None,None,"Need at least 2 years of data."
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(8)
                continue
            return None,None,None,None,None, f"Fetch error: {e}"
    else:
        return None,None,None,None,None,"Yahoo Finance unavailable. Please try again in a few minutes."

    try:
        def sg(df, keys, col=0):
            """Try multiple key names — yfinance field names vary by version."""
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                try:
                    v = df.loc[key].iloc[col]
                    f = float(v)
                    if f != 0.0 and str(v) != 'nan':
                        return f
                except Exception:
                    continue
            # last attempt: return 0
            for key in keys:
                try:
                    v = df.loc[key].iloc[col]
                    return float(v) if v is not None and str(v) != 'nan' else 0.0
                except Exception:
                    continue
            return 0.0

        r = dict(
            revt =sg(inc,['Total Revenue','Revenue']),
            cogs =sg(inc,['Cost Of Revenue','Cost Of Goods Sold','Reconciled Cost Of Revenue']),
            ni   =sg(inc,['Net Income','Net Income Common Stockholders']),
            xsga =sg(inc,['Selling General Administrative','Selling General And Administration','General And Administrative Expense']),
            at   =sg(bal,['Total Assets']),
            act  =sg(bal,['Current Assets','Total Current Assets']),
            lct  =sg(bal,['Current Liabilities','Total Current Liabilities']),
            lt   =sg(bal,['Total Liabilities Net Minority Interest','Total Liabilities']),
            rect =sg(bal,['Receivables','Net Receivables','Accounts Receivable']),
            ppent=sg(bal,['Net PPE','Net Property Plant And Equipment','Gross PPE']),
            dpc  =sg(cf, ['Depreciation And Amortization','Depreciation Amortization Depletion','Depreciation']),
            oancf=sg(cf, ['Operating Cash Flow','Cash Flow From Continuing Operating Activities']),
        )
        l = dict(
            revt =sg(inc,['Total Revenue','Revenue'],1),
            cogs =sg(inc,['Cost Of Revenue','Cost Of Goods Sold','Reconciled Cost Of Revenue'],1),
            ni   =sg(inc,['Net Income','Net Income Common Stockholders'],1),
            xsga =sg(inc,['Selling General Administrative','Selling General And Administration','General And Administrative Expense'],1),
            at   =sg(bal,['Total Assets'],1),
            act  =sg(bal,['Current Assets','Total Current Assets'],1),
            lt   =sg(bal,['Total Liabilities Net Minority Interest','Total Liabilities'],1),
            rect =sg(bal,['Receivables','Net Receivables','Accounts Receivable'],1),
            ppent=sg(bal,['Net PPE','Net Property Plant And Equipment','Gross PPE'],1),
            dpc  =sg(cf, ['Depreciation And Amortization','Depreciation Amortization Depletion','Depreciation'],1),
            oancf=sg(cf, ['Operating Cash Flow','Cash Flow From Continuing Operating Activities'],1),
        )

        sic_map = {
            'Technology':10,'Consumer Cyclical':50,'Healthcare':40,
            'Financial Services':60,'Energy':13,'Industrials':30,
            'Consumer Defensive':20,'Real Estate':65,'Utilities':49,
            'Communication Services':48,'Basic Materials':28
        }
        sic_est = sic_map.get(sector, 70)
        def map_ind(s):
            if 1000<=s<2000: return "Mining"
            elif 2000<=s<4000: return "Manufacturing"
            elif 4000<=s<5000: return "Transportation"
            elif 5000<=s<6000: return "Wholesale/Retail"
            elif 6000<=s<7000: return "Finance"
            elif 7000<=s<8000: return "Services"
            else: return "Other"
        industry = map_ind(sic_est)

        pm      = r['ni']/r['revt'] if r['revt'] else 0
        ind_avg = pm * 0.9

        feats = {
            'roa':                       r['ni']/r['at']           if r['at']   else 0,
            'profit_margin':             pm,
            'current_ratio':             r['act']/r['lct']         if r['lct']  else 0,
            'debt_ratio':                r['lt']/r['at']           if r['at']   else 0,
            'asset_turnover':            r['revt']/r['at']         if r['at']   else 0,
            'ocf_ratio':                 r['oancf']/r['at']        if r['at']   else 0,
            'sga_ratio':                 r['xsga']/r['revt']       if r['revt'] else 0,
            'depr_ratio':                r['dpc']/r['at']          if r['at']   else 0,
            'revenue_growth':            (r['revt']-l['revt'])/abs(l['revt']) if l['revt'] else 0,
            'asset_growth':              (r['at']-l['at'])/abs(l['at'])       if l['at']   else 0,
            'income_growth':             (r['ni']-l['ni'])/abs(l['ni'])       if l['ni']   else 0,
            'accrual_ratio':             (r['ni']-r['oancf'])/r['at']         if r['at']   else 0,
            'cfo_to_income':             r['oancf']/r['ni']        if r['ni']   else 0,
            'receivable_ratio':          r['rect']/r['revt']       if r['revt'] else 0,
            'profit_margin_vs_industry': pm - ind_avg,
        }

        m_score, _beneish_debug = compute_beneish(r, l)

        years = list(inc.columns)
        hist  = []
        for i, yr in enumerate(years):
            rv  = sg(inc,'Total Revenue',i)
            ni  = sg(inc,'Net Income',i)
            at  = sg(bal,'Total Assets',i)
            oc  = sg(cf, 'Operating Cash Flow',i)
            lc  = sg(bal,'Current Liabilities',i)
            ac  = sg(bal,'Current Assets',i)
            lt  = sg(bal,'Total Liabilities Net Minority Interest',i)
            hist.append({
                'Year':              pd.Timestamp(yr).year if hasattr(yr,'year') else str(yr)[:4],
                'Total Revenue':     rv/1e6,
                'Net Income':        ni/1e6,
                'Total Assets':      at/1e6,
                'Operating Cash Flow': oc/1e6,
                'ROA':               ni/at   if at  else 0,
                'Profit Margin':     ni/rv   if rv  else 0,
                'Debt Ratio':        lt/at   if at  else 0,
                'Current Ratio':     ac/lc   if lc  else 0,
                'Asset Turnover':    rv/at   if at  else 0,
                'Accrual Ratio':     (ni-oc)/at if at else 0,
            })
        hist_df = pd.DataFrame(hist).sort_values('Year')

        extra = dict(sector=sector, industry=industry,
                     country=country, employees=employees,
                     marketCap=0, beneish_debug=_beneish_debug)
        return feats, m_score, name, hist_df, extra

    except Exception as e:
        return None,None,None,None,None, f"Processing error: {e}"

def risk_label_ms(m):
    if m is None: return "Unknown","badge-mod"
    if m > -1.78: return "High Risk","badge-high"
    if m > -2.22: return "Moderate","badge-mod"
    return "Low Risk","badge-low"

def risk_label_prob(p):
    if p is None: return "N/A","badge-mod"
    if p > 0.65: return "High Risk","badge-high"
    if p > 0.40: return "Moderate","badge-mod"
    return "Low Risk","badge-low"

def fraud_score(m, p):
    ms = min(max((m + 2.5)/2.0, 0), 1)*50 if m is not None else 25
    ml = (p if p is not None else 0.3)*50
    return round(ms+ml, 1)

def count_flags(feats):
    n = 0
    if feats['accrual_ratio']            >  0.05: n+=1
    if feats['cfo_to_income']            <  0.50: n+=1
    if feats['receivable_ratio']         >  0.25: n+=1
    if feats['debt_ratio']               >  0.70: n+=1
    if feats['revenue_growth']           >  0.30: n+=1
    if feats['profit_margin_vs_industry']>  0.15: n+=1
    return n

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h1 style='font-size:1.8rem;font-weight:700;margin-bottom:2px;color:#fff'>🔍 FraudSight</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af;font-size:0.82rem;margin-top:-8px'>Financial Fraud Detection Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Company Ticker**")
    ticker_input = st.text_input("", placeholder="e.g. AAPL, MSFT, TSLA",
                                  label_visibility="collapsed",
                                  key="main_ticker").strip().upper()
    run_btn = st.button("▶  Run Analysis", use_container_width=True)
    st.markdown("---")
    page = st.radio("Navigate",
                    ["🏠 Home","📊 Company Analysis","📈 Financial Trends","⚖️ Compare Companies"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='color:#4a5568;font-size:0.78rem'>BA870AC820 · BU Questrom<br>Financial & Accounting Analytics</p>", unsafe_allow_html=True)

# ── Run Analysis ──────────────────────────────────────────────────────────────
if run_btn and ticker_input:
    fetch_company.clear()
    with st.spinner(f"Fetching data for {ticker_input}… (may take 10-20 seconds)"):
        feats, m_score, company_name, hist_df, extra, *err_parts = fetch_company(ticker_input)
        fetch_err = err_parts[0] if err_parts else None

    if feats is None:
        msg = fetch_err or "Unknown error."
        if any(x in msg.lower() for x in ["rate limit","too many","rate-limit"]):
            msg = msg + " Yahoo Finance rate limit hit. Please wait 60 seconds and try again."
        st.session_state["error"]  = msg
        st.session_state["result"] = None
    else:
        mdl = load_model()
        feat_df = pd.DataFrame([feats])
        prob        = float(mdl.predict_proba(feat_df[FEATURES])[0][1]) if mdl else None
        importances = mdl.feature_importances_ if mdl else None
        st.session_state['result'] = dict(
            ticker=ticker_input, name=company_name,
            feats=feats, m_score=m_score, prob=prob,
            hist_df=hist_df, extra=extra,
            importances=importances, score=fraud_score(m_score, prob),
        )
        st.session_state['error'] = None
elif run_btn and not ticker_input:
    st.session_state['error']  = "Please enter a ticker symbol first."
    st.session_state['result'] = None

res = st.session_state.get('result')
err = st.session_state.get('error')
mdl = load_model()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
<div style='text-align:center;padding:2.5rem 0 1.5rem 0'>
  <div style='font-family:"DM Serif Display",serif;font-size:3.2rem;font-weight:700;
              color:#0f1117;line-height:1.1;margin-bottom:0.6rem'>
    🔍 FraudSight
  </div>
  <div style='font-size:1.2rem;color:#6b7280;margin-bottom:0.3rem'>
    financial fraud detection for public companies
  
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, body in [
        (c1,"🎯","Our Mission","Democratize forensic financial analysis so investors, auditors, and analysts can quickly screen for manipulation risk using public data."),
        (c2,"👥","Who Is This For","Portfolio managers, equity analysts, credit risk teams, auditors, academic researchers, and students who need rapid, data-driven fraud screening."),
        (c3,"⚡","Why It Matters","The ACFE estimates global fraud losses exceed $4.7 trillion annually. Early detection protects investors and restores market trust."),
    ]:
        col.markdown(f"""
        <div class='fs-card'>
            <div style='font-size:1.6rem'>{icon}</div>
            <div style='font-weight:600;font-size:1rem;margin:8px 0 6px'>{title}</div>
            <div style='color:#6b7280;font-size:0.88rem;line-height:1.55'>{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    cl, cr = st.columns([1.1, 0.9])

    with cl:
        st.markdown("### 📦 Dataset & Features")
        st.markdown("""
<div class='fs-card'>
<p style='color:#374151;font-size:0.92rem;line-height:1.7'>
Our model was trained on <b>WRDS Compustat</b> annual financial statements covering
thousands of U.S. public companies. We engineered <b>15 ratio-based features</b>:
</p>
<table style='width:100%;font-size:0.86rem;border-collapse:collapse'>
<tr style='background:#f3f4f6'>
  <th style='padding:6px 10px;text-align:left'>Category</th>
  <th style='padding:6px 10px;text-align:left'>Features</th>
</tr>
<tr><td style='padding:5px 10px;border-top:1px solid #e5e7eb'>Profitability</td>
    <td style='padding:5px 10px;border-top:1px solid #e5e7eb'>ROA, Profit Margin</td></tr>
<tr style='background:#f9fafb'>
    <td style='padding:5px 10px;border-top:1px solid #e5e7eb'>Liquidity & Leverage</td>
    <td style='padding:5px 10px;border-top:1px solid #e5e7eb'>Current Ratio, Debt Ratio</td></tr>
<tr><td style='padding:5px 10px;border-top:1px solid #e5e7eb'>Efficiency</td>
    <td style='padding:5px 10px;border-top:1px solid #e5e7eb'>Asset Turnover, OCF Ratio, SG&A, Depr. Ratio</td></tr>
<tr style='background:#f9fafb'>
    <td style='padding:5px 10px;border-top:1px solid #e5e7eb'>Growth</td>
    <td style='padding:5px 10px;border-top:1px solid #e5e7eb'>Revenue, Asset, Income Growth</td></tr>
<tr><td style='padding:5px 10px;border-top:1px solid #e5e7eb'>Fraud Signals</td>
    <td style='padding:5px 10px;border-top:1px solid #e5e7eb'>Accrual Ratio, CFO/Income, Receivable Ratio, Margin vs Industry</td></tr>
</table>
</div>""", unsafe_allow_html=True)

    with cr:
        st.markdown("### 🧠 Models & Methods")
        st.markdown("""
<div class='fs-card'>
<p style='color:#374151;font-size:0.92rem;line-height:1.6;margin-bottom:14px'>
FraudSight combines two complementary detection methods:
</p>
<div style='border-left:3px solid #3b5bdb;padding-left:12px;margin-bottom:14px'>
<b>Beneish M-Score</b><br>
<span style='color:#6b7280;font-size:0.86rem'>
An 8-variable econometric model that detects earnings manipulation via accruals,
revenue patterns, and asset quality changes. Score > −1.78 signals risk.
</span>
</div>
<div style='border-left:3px solid #10b981;padding-left:12px'>
<b>Random Forest Classifier</b><br>
<span style='color:#6b7280;font-size:0.86rem'>
An ensemble ML model trained on 15 engineered features with
<code>class_weight='balanced'</code> to handle label imbalance.
Outputs a fraud probability (0–100%).
</span>
</div>
</div>""", unsafe_allow_html=True)

        st.markdown("### 🗺️ How to Use")
        st.markdown("""
<div class='fs-card'>
<ol style='color:#374151;font-size:0.88rem;line-height:2;padding-left:18px;margin:0'>
<li>Enter a <b>stock ticker</b> in the sidebar (e.g. <code>AAPL</code>)</li>
<li>Click <b>▶ Run Analysis</b></li>
<li>Navigate pages freely — results persist across all pages</li>
<li>For multi-company comparison, go to <b>⚖️ Compare Companies</b></li>
</ol>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – COMPANY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Company Analysis":
    st.markdown('<p class="page-title">Company Analysis</p>', unsafe_allow_html=True)

    if err:
        st.error(f"⚠️ {err}")
    elif res is None:
        st.info("👈 Enter a ticker in the sidebar and click **Run Analysis** to get started.")
    else:
        feats       = res['feats']
        m_score     = res['m_score']
        prob        = res['prob']
        score       = res['score']
        name        = res['name']
        extra       = res['extra']
        importances = res['importances']

        ms_label, ms_cls = risk_label_ms(m_score)
        ml_label, ml_cls = risk_label_prob(prob)

        # Header
        st.markdown(f"### {name} &nbsp;<span style='color:#6b7280;font-size:1rem;font-weight:400'>({res['ticker']})</span>", unsafe_allow_html=True)
        if isinstance(extra, dict):
            st.markdown(f"<p style='color:#6b7280;font-size:0.87rem'>"
                        f"{extra.get('sector','')} · {extra.get('country','')} · "
                        f"{extra.get('employees',0):,} employees</p>", unsafe_allow_html=True)

        # Score cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Fraud Risk Score",    f"{score:.1f} / 100",
                  help="Composite score: 0 = safe, 100 = high risk")
        k2.metric("Beneish M-Score",
                  f"{m_score:.3f}" if m_score is not None else "N/A",
                  help="Threshold: > −1.78 signals potential manipulation")
        k3.metric("ML Fraud Probability",
                  f"{prob:.1%}" if prob is not None else "N/A",
                  help="Random Forest probability")
        k4.metric("Risk Label", ms_label)

        # Show pkl warning if model not loaded
        if importances is None:
            st.warning("⚠️ **ML model not loaded** — `fraud_model.pkl` is missing from your GitHub repo. "
                       "Beneish M-Score is still shown. Upload the pkl file to enable ML predictions.")

        # Debug expander — shows raw fetched values so we can diagnose N/A
        if m_score is None and isinstance(extra, dict) and 'beneish_debug' in extra:
            with st.expander("🔎 Debug: Why is M-Score N/A? (click to expand)"):
                dbg = extra['beneish_debug']
                zero_keys = [k for k,v in dbg.items() if v == 0 and k.startswith('r_')]
                if zero_keys:
                    st.error(f"These values came back as 0 from yfinance: **{', '.join(zero_keys)}**")
                    st.write("This causes division-by-zero in the M-Score formula.")
                st.json(dbg)

        # Interpretation
        st.markdown("#### 📖 Score Interpretation")
        ic1, ic2 = st.columns(2)
        m_str = f"{m_score:.3f}" if m_score is not None else "N/A"
        with ic1:
            st.markdown(f"""
<div class='fs-card'>
<b>What is the Beneish M-Score?</b>
<p style='color:#6b7280;font-size:0.88rem;line-height:1.6;margin-top:6px'>
Developed by Prof. Messod Beneish (1999), the M-Score uses 8 financial ratios to detect
earnings manipulation. A score <b>above −1.78</b> suggests a company may be manipulating
its financials. {name}'s score of <b>{m_str}</b> places it in the
<b>{ms_label}</b> zone.
</p>
</div>""", unsafe_allow_html=True)
        with ic2:
            verdict = "⚠️ elevated caution recommended." if score > 60 else "✅ within typical range."
            st.markdown(f"""
<div class='fs-card'>
<b>What is the Fraud Risk Score?</b>
<p style='color:#6b7280;font-size:0.88rem;line-height:1.6;margin-top:6px'>
FraudSight's composite score (0–100) blends the M-Score signal with the ML fraud
probability. Scores above <b>60</b> warrant closer due diligence.
{name} scored <b>{score:.1f}/100</b> — {verdict}
</p>
</div>""", unsafe_allow_html=True)

        # Warning signs
        st.markdown("#### 🚩 Top Warning Signs")
        warnings = []
        if feats['accrual_ratio']            >  0.05: warnings.append(("Accrual Ratio",         f"{feats['accrual_ratio']:.3f}",          "Earnings driven by accruals rather than cash — a classic manipulation signal."))
        if feats['cfo_to_income']            <  0.50: warnings.append(("CFO / Net Income",       f"{feats['cfo_to_income']:.3f}",          "Operating cash flow is significantly lower than reported income."))
        if feats['receivable_ratio']         >  0.25: warnings.append(("Receivable Ratio",       f"{feats['receivable_ratio']:.3f}",       "High receivables relative to revenue may indicate inflated sales recognition."))
        if feats['debt_ratio']               >  0.70: warnings.append(("Debt Ratio",             f"{feats['debt_ratio']:.3f}",             "High leverage increases financial distress risk."))
        if feats['revenue_growth']           >  0.30: warnings.append(("Revenue Growth",         f"{feats['revenue_growth']:.1%}",         "Unusually high revenue growth can accompany aggressive accounting."))
        if feats['profit_margin_vs_industry']>  0.15: warnings.append(("Margin vs Industry",     f"{feats['profit_margin_vs_industry']:.3f}", "Profit margin significantly above peers may indicate overstated earnings."))

        if warnings:
            cols_w = st.columns(min(len(warnings), 3))
            for i, (lbl, val, desc) in enumerate(warnings):
                cols_w[i%3].markdown(f"""
<div style='background:#fff5f5;border:1px solid #fca5a5;border-left:4px solid #ef4444;
            border-radius:8px;padding:12px 14px;margin-bottom:10px'>
<div style='font-weight:600;color:#c0392b;font-size:0.88rem'>⚠️ {lbl}: {val}</div>
<div style='color:#7f1d1d;font-size:0.82rem;margin-top:4px'>{desc}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="fs-card" style="color:#166534">✅ No major warning signs detected.</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Feature importance + key ratios
        fa_col, kr_col = st.columns([1.1, 0.9])
        with fa_col:
            st.markdown("#### 🌲 Random Forest Feature Importance")
            if importances is not None:
                imp_df = pd.DataFrame({
                    'Feature': [FEATURE_LABELS[f] for f in FEATURES],
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(10)
                fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                 color='Importance',
                                 color_continuous_scale=[[0,'#bfdbfe'],[1,'#1d4ed8']],
                                 template='plotly_white')
                fig_imp.update_layout(height=380, showlegend=False,
                                      coloraxis_showscale=False,
                                      margin=dict(l=0,r=10,t=10,b=10))
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Place `fraud_model.pkl` in the app directory to enable ML scoring.")

        with kr_col:
            st.markdown("#### 📋 Key Ratio Summary")
            ratio_df = pd.DataFrame({
                'Ratio': [FEATURE_LABELS[f] for f in FEATURES],
                'Value': [f"{feats[f]:.4f}" for f in FEATURES]
            })
            st.dataframe(ratio_df, use_container_width=True, hide_index=True, height=380)

        # Industry comparison
        st.markdown("#### 🏭 Industry Comparison")
        industry = extra.get('industry','Other') if isinstance(extra, dict) else 'Other'
        bench    = IND_BENCH.get(industry, IND_BENCH['Other'])
        comp_keys= list(bench.keys())
        comp_lbl = [FEATURE_LABELS[k] for k in comp_keys]

        fig_ind = go.Figure()
        fig_ind.add_trace(go.Bar(name=name[:20],         x=comp_lbl, y=[feats[k] for k in comp_keys], marker_color='#3b5bdb'))
        fig_ind.add_trace(go.Bar(name=f'{industry} Avg', x=comp_lbl, y=[bench[k] for k in comp_keys], marker_color='#94a3b8'))
        fig_ind.update_layout(barmode='group', template='plotly_white', height=320,
                              margin=dict(l=0,r=0,t=10,b=80),
                              legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1))
        st.plotly_chart(fig_ind, use_container_width=True)
        st.caption(f"Industry benchmarks for **{industry}** are approximate sector averages.")

        # ── Beneish M-Score Deep Dive ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🧮 Beneish M-Score Deep Dive")
        st.markdown("""
<div class='fs-card'>
<p style='color:#374151;font-size:0.92rem;line-height:1.7;margin-bottom:10px'>
The <b>Beneish M-Score</b> was developed by Prof. Messod Beneish (Indiana University, 1999)
as an econometric model to detect earnings manipulation. It uses <b>8 financial ratios</b>
derived from two consecutive years of financial statements. The final score is computed as:
</p>
<div style='background:#f3f4f6;border-radius:8px;padding:12px 16px;font-family:monospace;font-size:0.85rem;margin-bottom:12px'>
M = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI
</div>
<p style='color:#6b7280;font-size:0.87rem'>
<b>Interpretation:</b> A score <b style="color:#c0392b">above −1.78</b> suggests the company
is likely manipulating earnings. A score <b style="color:#1a7a30">below −2.22</b> indicates
low manipulation risk. The range between is considered a gray zone.
</p>
</div>""", unsafe_allow_html=True)

        # Compute 8 Beneish ratios directly from feats
        def sd(a, b, d=1.0): return a/b if b and b != 0 else d

        f = feats
        # Reconstruct raw values from derived ratios where possible
        # receivable_ratio = rect/revt  →  use as proxy for current rect/revt
        # accrual_ratio    = (ni-oancf)/at
        # cfo_to_income    = oancf/ni
        # We compute each index using what we have

        # DSRI: need rect/revt for t and t-1; use receivable_ratio as t, estimate t-1
        dsri_t  = f['receivable_ratio']                      # rect_t / revt_t
        # approximate t-1 ratio: if revenue grew, receivables may have grown similarly
        rev_g   = f['revenue_growth']  # (revt_t - revt_l)/revt_l
        inc_g   = f['income_growth']
        ast_g   = f['asset_growth']
        # reconstruct t-1 proxies
        dsri_l  = dsri_t / (1 + (rev_g * 0.5)) if rev_g > -1 else dsri_t
        DSRI_v  = sd(dsri_t, dsri_l)

        # GMI: gross margin t-1 / gross margin t  (sga_ratio proxy not perfect, use profit_margin)
        pm_t    = f['profit_margin']
        pm_l    = pm_t / (1 + inc_g) if inc_g > -1 else pm_t
        GMI_v   = sd(pm_l, pm_t)

        # AQI: change in non-operating assets ratio
        # (1 - (act+ppe)/at) for t and t-1 — approximate using asset_growth
        act_at_t = sd(f['current_ratio'] * f['debt_ratio'] * 0.5, 1, 0.3)  # rough proxy
        AQI_v   = 1.0  # neutral default when raw data unavailable

        # SGI: revt / revt_l = 1 + revenue_growth
        SGI_v   = 1 + rev_g

        # DEPI: dep_rate_l / dep_rate_t — use depr_ratio directly
        dep_t   = f['depr_ratio']
        dep_l   = dep_t / (1 + ast_g) if ast_g > -1 else dep_t
        DEPI_v  = sd(dep_l, dep_t)

        # SGAI: (sga/rev)_t / (sga/rev)_l
        sga_t   = f['sga_ratio']
        sga_l   = sga_t / (1 + rev_g * 0.8) if rev_g > -1 else sga_t
        SGAI_v  = sd(sga_t, sga_l)

        # TATA: (ni - oancf) / at  =  accrual_ratio directly
        TATA_v  = f['accrual_ratio']

        # LVGI: (lt/at)_t / (lt/at)_l
        dr_t    = f['debt_ratio']
        dr_l    = dr_t / (1 + ast_g * 0.6) if ast_g > -1 else dr_t
        LVGI_v  = sd(dr_t, dr_l)

        # Override with debug values if they exist (more accurate)
        r_d = res['extra'].get('beneish_debug', {}) if isinstance(res['extra'], dict) else {}
        if r_d.get('DSRI') is not None: DSRI_v = r_d['DSRI']
        if r_d.get('GMI')  is not None: GMI_v  = r_d['GMI']
        if r_d.get('AQI')  is not None: AQI_v  = r_d['AQI']
        if r_d.get('SGI')  is not None: SGI_v  = r_d['SGI']
        if r_d.get('DEPI') is not None: DEPI_v = r_d['DEPI']
        if r_d.get('SGAI') is not None: SGAI_v = r_d['SGAI']
        if r_d.get('TATA') is not None: TATA_v = r_d['TATA']
        if r_d.get('LVGI') is not None: LVGI_v = r_d['LVGI']

        BENEISH_RATIOS = [
            {
                "name": "DSRI",
                "full": "Days Sales in Receivables Index",
                "formula": "(Receivables / Revenue)ₜ ÷ (Receivables / Revenue)ₜ₋₁",
                "value": DSRI_v,
                "what": "Measures whether receivables are growing faster than revenue. A high DSRI suggests the company may be recording revenue before actually collecting cash — a common manipulation tactic.",
                "threshold": "Normal ≈ 1.0 · Concern > 1.3",
                "concern": "high",
            },
            {
                "name": "GMI",
                "full": "Gross Margin Index",
                "formula": "Gross Margin ₜ₋₁ ÷ Gross Margin ₜ",
                "value": GMI_v,
                "what": "Compares gross margin between periods. A value > 1 means gross margin deteriorated — companies with worsening margins may be more tempted to manipulate earnings.",
                "threshold": "Normal ≈ 1.0 · Concern > 1.2",
                "concern": "high",
            },
            {
                "name": "AQI",
                "full": "Asset Quality Index",
                "formula": "(1 − (Current Assets + PPE) / Total Assets)ₜ ÷ same ₜ₋₁",
                "value": AQI_v,
                "what": "Measures changes in intangible or deferred assets relative to total assets. A high AQI suggests the company is capitalizing more costs (turning expenses into assets), which inflates earnings.",
                "threshold": "Normal ≈ 1.0 · Concern > 1.25",
                "concern": "high",
            },
            {
                "name": "SGI",
                "full": "Sales Growth Index",
                "formula": "Revenue ₜ ÷ Revenue ₜ₋₁",
                "value": SGI_v,
                "what": "Measures revenue growth. High growth companies face more pressure to meet expectations and are statistically more likely to manipulate earnings.",
                "threshold": "Normal ≈ 1.0–1.1 · Concern > 1.6",
                "concern": "high",
            },
            {
                "name": "DEPI",
                "full": "Depreciation Index",
                "formula": "Depreciation Rate ₜ₋₁ ÷ Depreciation Rate ₜ",
                "value": DEPI_v,
                "what": "Detects whether the company is slowing down its depreciation rate, which would boost reported earnings by spreading asset costs over a longer period.",
                "threshold": "Normal ≈ 1.0 · Concern > 1.1",
                "concern": "high",
            },
            {
                "name": "SGAI",
                "full": "SG&A Expense Index",
                "formula": "(SG&A / Revenue)ₜ ÷ (SG&A / Revenue)ₜ₋₁",
                "value": SGAI_v,
                "what": "Tracks whether selling, general & administrative expenses are growing relative to revenue. Rising SGAI may indicate operational inefficiency or undisclosed costs.",
                "threshold": "Normal ≈ 1.0 · Concern > 1.1",
                "concern": "high",
            },
            {
                "name": "TATA",
                "full": "Total Accruals to Total Assets",
                "formula": "(Net Income − Operating Cash Flow) ÷ Total Assets",
                "value": TATA_v,
                "what": "The most important fraud signal. High accruals mean earnings are not backed by actual cash flow — a hallmark of earnings manipulation. Legitimate profits should be supported by real cash.",
                "threshold": "Normal < 0.05 · Concern > 0.10",
                "concern": "high",
            },
            {
                "name": "LVGI",
                "full": "Leverage Index",
                "formula": "(Total Liabilities / Assets)ₜ ÷ (Total Liabilities / Assets)ₜ₋₁",
                "value": LVGI_v,
                "what": "Tracks changes in financial leverage. Increasing leverage raises the risk of debt covenant violations, which can motivate management to manipulate earnings upward.",
                "threshold": "Normal ≈ 1.0 · Concern > 1.2",
                "concern": "high",
            },
        ]

        st.markdown("##### The 8 Beneish Ratios")
        for ratio in BENEISH_RATIOS:
            val = ratio['value']
            val_str = f"{val:.4f}" if val is not None else "N/A (insufficient data)"
            with st.expander(f"**{ratio['name']}** — {ratio['full']}"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
<div style='background:#f8fafc;border-radius:10px;padding:14px;text-align:center'>
  <div style='font-size:1.6rem;font-weight:700;color:#1d4ed8'>{val_str}</div>
  <div style='color:#6b7280;font-size:0.78rem;margin-top:4px'>{ratio['threshold']}</div>
</div>
<div style='margin-top:10px;background:#f3f4f6;border-radius:8px;padding:10px;font-size:0.8rem;font-family:monospace;color:#374151'>
{ratio['formula']}
</div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
<div style='padding:6px 0'>
  <p style='color:#374151;font-size:0.9rem;line-height:1.65;margin:0'>{ratio['what']}</p>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – FINANCIAL TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Financial Trends":
    st.markdown('<p class="page-title">Financial Trends</p>', unsafe_allow_html=True)

    if err:
        st.error(f"⚠️ {err}")
    elif res is None:
        st.info("👈 Enter a ticker in the sidebar and click **Run Analysis** first.")
    else:
        hist_df = res['hist_df']
        name    = res['name']
        st.markdown(f"Viewing historical trends for **{name}** ({res['ticker']})")

        st.markdown("#### Select Metrics to Visualize")
        available = [m for m in TREND_METRICS if m in hist_df.columns]
        selected  = st.multiselect(
            "Choose one or more metrics:",
            options=available,
            default=available[:3],
            format_func=lambda x: TREND_METRICS[x]
        )
        show_btn = st.button("📊 Show Trends", use_container_width=False)

        if show_btn or selected:
            if not selected:
                st.warning("Please select at least one metric.")
            else:
                st.markdown("#### 📉 Trend Chart")

                # Separate dollar-scale and ratio-scale metrics
                dollar_metrics = {'Total Revenue','Net Income','Total Assets','Operating Cash Flow'}
                dollar_sel = [m for m in selected if m in dollar_metrics]
                ratio_sel  = [m for m in selected if m not in dollar_metrics]

                if dollar_sel and ratio_sel:
                    fig_t = make_subplots(specs=[[{"secondary_y": True}]])
                    for i, m in enumerate(dollar_sel):
                        fig_t.add_trace(go.Scatter(
                            x=hist_df['Year'].astype(str), y=hist_df[m],
                            mode='lines+markers', name=TREND_METRICS[m],
                            line=dict(color=PALETTE[i], width=2.5), marker=dict(size=7)
                        ), secondary_y=False)
                    for i, m in enumerate(ratio_sel):
                        fig_t.add_trace(go.Scatter(
                            x=hist_df['Year'].astype(str), y=hist_df[m],
                            mode='lines+markers', name=TREND_METRICS[m],
                            line=dict(color=PALETTE[len(dollar_sel)+i], width=2.5, dash='dash'),
                            marker=dict(size=7)
                        ), secondary_y=True)
                    fig_t.update_yaxes(title_text="Value ($M)", secondary_y=False)
                    fig_t.update_yaxes(title_text="Ratio",      secondary_y=True)
                else:
                    fig_t = go.Figure()
                    for i, m in enumerate(selected):
                        fig_t.add_trace(go.Scatter(
                            x=hist_df['Year'].astype(str), y=hist_df[m],
                            mode='lines+markers', name=TREND_METRICS[m],
                            line=dict(color=PALETTE[i], width=2.5), marker=dict(size=7)
                        ))

                fig_t.update_layout(
                    template='plotly_white', height=380,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=0,r=0,t=30,b=10),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_t, use_container_width=True)

                st.markdown("#### 📖 Metric Interpretations")
                for m in selected:
                    vals = hist_df[m].tolist()
                    direction = "📈 increasing" if len(vals) > 1 and vals[-1] > vals[0] else "📉 decreasing"
                    st.markdown(f"""
<div class='fs-card'>
<b>{TREND_METRICS[m]}</b>
&nbsp;&nbsp;<span style='color:#6b7280;font-size:0.83rem'>{direction} over the period</span>
<p style='color:#374151;font-size:0.88rem;line-height:1.6;margin-top:6px'>
{METRIC_INTERP.get(m,'')}
</p>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – COMPARE COMPANIES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Compare Companies":
    st.markdown('<p class="page-title">Compare Companies</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Compare fraud risk and financial ratios across up to 10 companies</p>', unsafe_allow_html=True)

    default_anchor = res['ticker'] if res else ""

    st.markdown("#### Enter Tickers to Compare")
    st.caption("Add 2–10 tickers separated by commas. Example: AAPL, MSFT, GOOGL")
    comp_input = st.text_input(
        "Tickers:",
        placeholder="e.g. AAPL, MSFT, GOOGL",
        value=default_anchor,
        label_visibility="collapsed"
    )
    comp_btn = st.button("⚖️ Run Comparison", use_container_width=False)

    if comp_btn:
        raw = [t.strip().upper() for t in comp_input.split(',') if t.strip()]
        raw = list(dict.fromkeys(raw))

        if len(raw) < 1:
            st.markdown('<div class="warn-banner">⛔ Please enter at least one ticker.</div>', unsafe_allow_html=True)
        elif len(raw) == 1:
            st.markdown('<div class="warn-banner">⚠️ Please add more companies (2–10 tickers) to enable comparison.</div>', unsafe_allow_html=True)
        elif len(raw) > 10:
            st.markdown('<div class="warn-banner">⛔ Maximum 10 companies allowed. Please remove some tickers.</div>', unsafe_allow_html=True)
        else:
            rows = []
            prog = st.progress(0, text="Fetching data…")
            for i, tk in enumerate(raw):
                prog.progress((i+1)/len(raw), text=f"Fetching {tk}…")
                feats, m_score, cname, hist_df, extra, *err_p = fetch_company(tk)
                fetch_err = err_p[0] if err_p else None
                if feats is None:
                    st.warning(f"⚠️ Could not fetch **{tk}**: {fetch_err}")
                    continue
                prob = float(mdl.predict_proba(pd.DataFrame([feats])[FEATURES])[0][1]) if mdl else None
                ml_label, _ = risk_label_prob(prob)
                ms_label, _ = risk_label_ms(m_score)
                rows.append(dict(
                    ticker=tk, name=cname[:22],
                    feats=feats, m_score=m_score, prob=prob,
                    score=fraud_score(m_score,prob),
                    ms_label=ms_label, ml_label=ml_label,
                ))
            prog.empty()

            if len(rows) < 2:
                st.error("Not enough valid companies to compare. Please check your tickers.")
            else:
                names   = [r['name']   for r in rows]
                scores  = [r['score']  for r in rows]
                mscores = [r['m_score'] or 0 for r in rows]
                probs   = [r['prob']   or 0  for r in rows]
                flags   = [count_flags(r['feats']) for r in rows]

                # Summary table
                st.markdown("#### 📋 Summary")
                st.dataframe(pd.DataFrame({
                    'Company':    names,
                    'Ticker':     [r['ticker']   for r in rows],
                    'Fraud Score':[f"{s:.1f}"    for s in scores],
                    'M-Score':    [f"{m:.3f}"    for m in mscores],
                    'ML Prob':    [f"{p:.1%}"    for p in probs],
                    'Risk':       [r['ms_label'] for r in rows],
                    'Red Flags':  flags,
                }), hide_index=True, use_container_width=True)

                # Fraud score bar
                st.markdown("#### 🎯 Fraud Risk Score")
                st.caption("Composite score (0–100) combining Beneish M-Score and ML fraud probability. The red dashed line marks the high-risk threshold at 60. Companies above this line warrant further due diligence.")
                fig_s = go.Figure(go.Bar(
                    x=names, y=scores,
                    marker_color=PALETTE[:len(rows)],
                    text=[f"{s:.1f}" for s in scores], textposition='outside'
                ))
                fig_s.add_hline(y=60, line_dash='dash', line_color='red',
                                annotation_text='High Risk (60)')
                fig_s.update_layout(template='plotly_white', height=320,
                                    yaxis_range=[0,108], margin=dict(l=0,r=0,t=10,b=10))
                st.plotly_chart(fig_s, use_container_width=True)

                # Key ratio tabs
                st.markdown("#### 📊 Key Ratio Comparisons")
                st.caption("Select a tab to compare each financial ratio across companies. Higher ROA and Profit Margin indicate better profitability. Lower Debt Ratio is safer. Accrual Ratio and Receivable Ratio closer to 0 are healthier — high values may indicate earnings manipulation.")
                ratio_keys = ['roa','profit_margin','debt_ratio','current_ratio','accrual_ratio','receivable_ratio']
                ratio_interp = {
                    'roa':              "**Return on Assets (ROA):** Measures how efficiently a company generates profit from its assets. Higher is better. Significant differences across peers suggest varying operational efficiency.",
                    'profit_margin':    "**Profit Margin:** The percentage of revenue that becomes profit. A company with a much higher margin than peers may be reporting inflated earnings — worth investigating.",
                    'debt_ratio':       "**Debt Ratio:** Total liabilities divided by total assets. Values above 0.7 indicate high leverage and elevated financial risk.",
                    'current_ratio':    "**Current Ratio:** Current assets divided by current liabilities. Values below 1.0 suggest the company may struggle to meet short-term obligations.",
                    'accrual_ratio':    "**Accrual Ratio:** (Net Income − Operating CFO) / Assets. A high positive value means earnings are driven by accruals rather than actual cash — a key fraud signal.",
                    'receivable_ratio': "**Receivable Ratio:** Accounts receivable relative to revenue. A rising ratio may indicate the company is booking sales that haven't been collected yet.",
                }
                r_tabs = st.tabs([FEATURE_LABELS[k].split('(')[0].strip() for k in ratio_keys])
                for tab, rk in zip(r_tabs, ratio_keys):
                    with tab:
                        st.markdown(ratio_interp.get(rk,''))
                        vals = [r['feats'][rk] for r in rows]
                        fig_r = go.Figure(go.Bar(
                            x=names, y=vals, marker_color=PALETTE[:len(rows)],
                            text=[f"{v:.3f}" for v in vals], textposition='outside'
                        ))
                        fig_r.update_layout(template='plotly_white', height=300,
                                            margin=dict(l=0,r=0,t=10,b=10))
                        st.plotly_chart(fig_r, use_container_width=True)

                # Red flags bar
                st.markdown("#### 🚩 Red Flag Count")
                st.caption("Number of fraud warning signals triggered per company (max 6). 🟢 Green = 0 flags (clean), 🟡 Orange = 1–2 flags (monitor), 🔴 Red = 3+ flags (high concern). Flags are based on accrual ratio, CFO/income gap, receivables, debt level, revenue growth, and margin vs. industry.")
                fig_f = go.Figure(go.Bar(
                    x=names, y=flags,
                    marker_color=['#ef4444' if f>=3 else '#f59e0b' if f>=1 else '#10b981' for f in flags],
                    text=flags, textposition='outside'
                ))
                fig_f.update_layout(template='plotly_white', height=300,
                                    yaxis_range=[0, max(flags)+1.5],
                                    margin=dict(l=0,r=0,t=10,b=10))
                st.plotly_chart(fig_f, use_container_width=True)

                # Radar
                st.markdown("#### 🕸️ Risk Profile Radar")
                st.caption("Each axis represents a normalized financial metric (0–1 scale). A larger, rounder shape generally indicates stronger overall financial health. Differences in shape between companies highlight where each firm stands out or falls behind its peers.")
                radar_keys   = ['roa','profit_margin','current_ratio','asset_turnover','ocf_ratio','accrual_ratio']
                radar_labels = [FEATURE_LABELS[k].split('(')[0].strip() for k in radar_keys] + [FEATURE_LABELS[radar_keys[0]].split('(')[0].strip()]
                fig_rad = go.Figure()
                for i, r in enumerate(rows):
                    vals = [min(max((r['feats'][k]+0.5)/1.0,0),1) for k in radar_keys]
                    vals += [vals[0]]
                    fig_rad.add_trace(go.Scatterpolar(
                        r=vals, theta=radar_labels,
                        fill='toself', name=r['name'],
                        line_color=PALETTE[i%len(PALETTE)], opacity=0.75
                    ))
                fig_rad.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                    showlegend=True, height=440,
                    margin=dict(l=30,r=30,t=30,b=30)
                )
                st.plotly_chart(fig_rad, use_container_width=True)
