import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Trend & Survey Dashboard", layout="wide")
st.title("📈 Stock Market Trend Visualization & Survey Dashboard")

tab1, tab2, tab3 = st.tabs(["Stock MA Dashboard", "Survey Analytics", "User Adoption Predictor"])

# ---------------- TAB 1: Advanced Quantitative Market Terminal ----------------

with tab1:
    st.markdown("### 📈 Quantitative Market Terminal & Indicator Engine")
    st.caption("Real-time technical indicator computation, crossover detection, and multi-factor trend momentum.")

    # Top Control Bar inside an elevated Card
    with st.container(border=True):
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.5, 1.2, 1.2, 2.1])
        
        with ctrl1:
            ticker = st.selectbox(
                "🏷️ Asset / Ticker",
                ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "MSFT", "GOOGL", "NVDA"],
                index=0
            )
        with ctrl2:
            period = st.selectbox(
                "📅 Interval Horizon",
                ["3mo", "6mo", "1y", "2y", "5y", "max"],
                index=2
            )
        with ctrl3:
            chart_type = st.selectbox("📊 Chart Archetype", ["Candlestick", "Line (Close)"])
        
        with ctrl4:
            selected_indicators = st.multiselect(
                "⚡ Overlay Technicals",
                options=["SMA 20 & 50", "EMA 20 & 50", "Bollinger Bands", "RSI (14)", "MACD"],
                default=["SMA 20 & 50", "RSI (14)"]
            )

    # Data Ingestion & Transformation
    with st.spinner(f"Ingesting live ticks for {ticker}..."):
        try:
            stock = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if stock.empty:
                t_obj = yf.Ticker(ticker)
                stock = t_obj.history(period=period)
        except Exception as e:
            stock = pd.DataFrame()

    if not stock.empty:
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = [col[0] for col in stock.columns]

        # 1. Moving Averages
        stock['SMA_20'] = stock['Close'].rolling(window=20).mean()
        stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
        stock['EMA_20'] = stock['Close'].ewm(span=20, adjust=False).mean()
        stock['EMA_50'] = stock['Close'].ewm(span=50, adjust=False).mean()

        # 2. Bollinger Bands
        bb_std = stock['Close'].rolling(window=20).std()
        stock['BB_Upper'] = stock['SMA_20'] + (bb_std * 2)
        stock['BB_Lower'] = stock['SMA_20'] - (bb_std * 2)

        # 3. Relative Strength Index (RSI 14)
        delta = stock['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        stock['RSI_14'] = 100 - (100 / (1 + rs))

        # 4. MACD (12, 26, 9)
        ema_12 = stock['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = stock['Close'].ewm(span=26, adjust=False).mean()
        stock['MACD'] = ema_12 - ema_26
        stock['MACD_Signal'] = stock['MACD'].ewm(span=9, adjust=False).mean()
        stock['MACD_Hist'] = stock['MACD'] - stock['MACD_Signal']

        # Executive Performance HUD
        latest_close = float(stock['Close'].iloc[-1])
        prev_close = float(stock['Close'].iloc[-2])
        price_change = latest_close - prev_close
        pct_change = (price_change / prev_close) * 100
        period_high = float(stock['High'].max())
        period_low = float(stock['Low'].min())
        latest_vol = int(stock['Volume'].iloc[-1])

        hud_c1, hud_c2, hud_c3, hud_c4 = st.columns(4)
        with hud_c1:
            st.metric(label=f"{ticker} Spot Price", value=f"{latest_close:,.2f}", delta=f"{price_change:+,.2f} ({pct_change:+.2f}%)")
        with hud_c2:
            st.metric(label="Period High / Low", value=f"{period_high:,.1f}", delta=f"Low: {period_low:,.1f}", delta_color="off")
        with hud_c3:
            rsi_val = stock['RSI_14'].iloc[-1]
            rsi_state = "Overbought" if rsi_val >= 70 else "Oversold" if rsi_val <= 30 else "Neutral"
            st.metric(label="RSI (14) Momentum", value=f"{rsi_val:.1f}", delta=rsi_state, delta_color="inverse" if rsi_val >= 70 else "normal")
        with hud_c4:
            sma_diff = stock['SMA_20'].iloc[-1] - stock['SMA_50'].iloc[-1]
            trend_signal = "🟢 Bullish (SMA20 > 50)" if sma_diff > 0 else "🔴 Bearish (SMA20 < 50)"
            st.metric(label="Moving Avg Alignment", value=trend_signal)

        # Plotly Subplots
        has_rsi = "RSI (14)" in selected_indicators
        has_macd = "MACD" in selected_indicators

        rows = 2 + (1 if has_rsi else 0) + (1 if has_macd else 0)
        row_heights = [0.55, 0.15] + ([0.15] if has_rsi else []) + ([0.15] if has_macd else [])
        subplot_titles = [f"{ticker} Price Action", "Volume"]
        if has_rsi: subplot_titles.append("Relative Strength Index (RSI)")
        if has_macd: subplot_titles.append("MACD Oscillator")

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles
        )

        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=stock.index, open=stock['Open'], high=stock['High'],
                low=stock['Low'], close=stock['Close'], name="OHLC",
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=stock.index, y=stock['Close'], mode='lines', name='Close Price',
                line=dict(color='#2962ff', width=2)
            ), row=1, col=1)

        if "SMA 20 & 50" in selected_indicators:
            fig.add_trace(go.Scatter(x=stock.index, y=stock['SMA_20'], line=dict(color='#ff9800', width=1.5), name='SMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock.index, y=stock['SMA_50'], line=dict(color='#4caf50', width=1.5), name='SMA 50'), row=1, col=1)

        if "EMA 20 & 50" in selected_indicators:
            fig.add_trace(go.Scatter(x=stock.index, y=stock['EMA_20'], line=dict(color='#e91e63', width=1.5, dash='dot'), name='EMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock.index, y=stock['EMA_50'], line=dict(color='#9c27b0', width=1.5, dash='dot'), name='EMA 50'), row=1, col=1)

        if "Bollinger Bands" in selected_indicators:
            fig.add_trace(go.Scatter(x=stock.index, y=stock['BB_Upper'], line=dict(color='rgba(150, 150, 150, 0.4)', width=1), name='BB Upper', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock.index, y=stock['BB_Lower'], line=dict(color='rgba(150, 150, 150, 0.4)', width=1), fill='tonexty', fillcolor='rgba(100, 100, 100, 0.08)', name='BB Band'), row=1, col=1)

        vol_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(stock['Close'], stock['Open'])]
        fig.add_trace(go.Bar(
            x=stock.index, y=stock['Volume'], name='Volume',
            marker_color=vol_colors, showlegend=False
        ), row=2, col=1)

        current_row = 3
        if has_rsi:
            fig.add_trace(go.Scatter(x=stock.index, y=stock['RSI_14'], line=dict(color='#ab47bc', width=1.8), name='RSI 14'), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            current_row += 1

        if has_macd:
            fig.add_trace(go.Scatter(x=stock.index, y=stock['MACD'], line=dict(color='#29b6f6', width=1.5), name='MACD'), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=stock.index, y=stock['MACD_Signal'], line=dict(color='#ff7043', width=1.5), name='Signal'), row=current_row, col=1)
            hist_colors = ['#26a69a' if val >= 0 else '#ef5350' for val in stock['MACD_Hist']]
            fig.add_trace(go.Bar(x=stock.index, y=stock['MACD_Hist'], marker_color=hist_colors, name='MACD Hist', showlegend=False), row=current_row, col=1)

        fig.update_layout(
            height=750,
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.error("⚠️ Failed to fetch market telemetry. Check internet connection or ticker syntax.")
        
# ---------------- TAB 2: Survey Data Visualizations (Advanced Analytics Hub) ----------------


with tab2:
    st.markdown("### 📊 Empirical Survey Intelligence & Cohort Analytics")
    st.caption("Multivariate descriptive analytics, demographic distributions, and sentiment matrices across respondent cohorts.")

    df_survey = pd.read_csv("data/stock_market_survey_updated.csv")
    exp_col = df_survey.columns[7]

    # --- Cohort Filter Control Bar ---
    with st.container(border=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            gender_filter = st.multiselect(
                "Filter by Gender",
                options=df_survey['Gender'].dropna().unique(),
                default=list(df_survey['Gender'].dropna().unique())
            )
        with f_col2:
            occ_col = 'Occupation' if 'Occupation' in df_survey.columns else df_survey.columns[3]
            occ_filter = st.multiselect(
                "Filter by Occupation",
                options=df_survey[occ_col].dropna().unique(),
                default=list(df_survey[occ_col].dropna().unique())
            )
        with f_col3:
            age_range = st.slider(
                "Filter Age Demographic",
                int(df_survey['Age'].min()),
                int(df_survey['Age'].max()),
                (int(df_survey['Age'].min()), int(df_survey['Age'].max()))
            )

    # Filter dataframe
    filtered_df = df_survey[
        (df_survey['Gender'].isin(gender_filter)) &
        (df_survey[occ_col].isin(occ_filter)) &
        (df_survey['Age'].between(age_range[0], age_range[1]))
    ]

    if filtered_df.empty:
        st.warning("⚠️ No survey records match the active filter criteria. Adjust demographic sliders.")
    else:
        # --- Top Executive KPIs ---
        total_sample = len(filtered_df)
        useful_yes = (filtered_df['Dashboard_Useful'].str.strip().str.lower() == 'yes').sum()
        satisfaction_rate = (useful_yes / total_sample) * 100 if total_sample > 0 else 0
        median_age = filtered_df['Age'].median()
        top_pref = filtered_df['Investment_Interest'].mode()[0] if not filtered_df['Investment_Interest'].empty else "N/A"

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Survey Sample Size", f"{total_sample} Responses")
        kpi2.metric("Median Cohort Age", f"{median_age:.0f} Yrs")
        kpi3.metric("Dominant Asset", f"{top_pref}")
        kpi4.metric("Dashboard Approval", f"{satisfaction_rate:.1f}%", delta="Positive Sentiment")

        st.divider()

        # Shared Plotly Dark Theme Preset
        chart_layout = dict(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=35, b=10),
            height=320
        )

        # --- Row 1: Demographics (Age Distribution & Gender Donut) ---
        r1_a, r1_b = st.columns(2)
        
        with r1_a:
            with st.container(border=True):
                st.markdown("##### 1. Age Cohort Distribution")
                age_counts = filtered_df['Age'].value_counts().sort_index().reset_index()
                age_counts.columns = ['Age', 'Count']
                
                fig_age = go.Figure(go.Bar(
                    x=age_counts['Age'],
                    y=age_counts['Count'],
                    marker=dict(color=age_counts['Count'], colorscale='Blues'),
                    text=age_counts['Count'],
                    textposition='outside'
                ))
                fig_age.update_layout(**chart_layout, showlegend=False, xaxis_title="Age", yaxis_title="Respondents")
                st.plotly_chart(fig_age, width='stretch')

        with r1_b:
            with st.container(border=True):
                st.markdown("##### 2. Gender Composition")
                gender_counts = filtered_df['Gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']
                
                fig_gender = go.Figure(go.Pie(
                    labels=gender_counts['Gender'],
                    values=gender_counts['Count'],
                    hole=0.6,
                    marker=dict(colors=['#00b4d8', '#ff758f', '#90e0ef']),
                    textinfo='label+percent'
                ))
                fig_gender.update_layout(**chart_layout, showlegend=True, legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig_gender, width='stretch')

        # --- Row 2: Investment Preferences & Market Experience ---
        r2_a, r2_b = st.columns(2)

        with r2_a:
            with st.container(border=True):
                st.markdown("##### 3. Preferred Investment Asset Class")
                pref_counts = filtered_df['Investment_Interest'].value_counts().reset_index()
                pref_counts.columns = ['Asset', 'Count']
                
                fig_pref = go.Figure(go.Bar(
                    x=pref_counts['Count'],
                    y=pref_counts['Asset'],
                    orientation='h',
                    marker=dict(color=pref_counts['Count'], colorscale='Viridis'),
                    text=pref_counts['Count'],
                    textposition='outside'
                ))
                fig_pref.update_layout(**chart_layout, showlegend=False, yaxis=dict(autorange="reversed"), xaxis_title="Count")
                st.plotly_chart(fig_pref, width='stretch')

        with r2_b:
            with st.container(border=True):
                st.markdown("##### 4. Stock Market Experience Tenure")
                exp_counts = filtered_df[exp_col].value_counts().reset_index() 
                exp_counts.columns = ['Tenure', 'Count']
                
                fig_exp = go.Figure(go.Pie(
                    labels=exp_counts['Tenure'],
                    values=exp_counts['Count'],
                    hole=0.55,
                    marker=dict(colors=['#2a9d8f', '#e76f51', '#f4a261', '#e9c46a']),
                    textinfo='label+percent'
                ))
                fig_exp.update_layout(**chart_layout, showlegend=True, legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig_exp, width='stretch')

        # --- Row 3: Technical Resources & Perceived Adoption ---
        r3_a, r3_b = st.columns(2)

        with r3_a:
            with st.container(border=True):
                st.markdown("##### 5. Most Valued Technical Feature")
                src_counts = filtered_df['Most_Useful_Information'].value_counts().reset_index()
                src_counts.columns = ['Feature', 'Count']
                
                fig_src = go.Figure(go.Bar(
                    x=src_counts['Count'],
                    y=src_counts['Feature'],
                    orientation='h',
                    marker=dict(color=src_counts['Count'], colorscale='Sunset'),
                    text=src_counts['Count'],
                    textposition='outside'
                ))
                fig_src.update_layout(**chart_layout, showlegend=False, yaxis=dict(autorange="reversed"), xaxis_title="Count")
                st.plotly_chart(fig_src, width='stretch')

        with r3_b:
            with st.container(border=True):
                st.markdown("##### 6. Dashboard Usefulness Sentiment")
                dash_counts = filtered_df['Dashboard_Useful'].value_counts().reset_index()
                dash_counts.columns = ['Sentiment', 'Count']
                
                sentiment_colors = {'Yes': '#2ec4b6', 'Maybe': '#ffbf69', 'No': '#e71d36'}
                bar_colors = [sentiment_colors.get(s, '#3a86ff') for s in dash_counts['Sentiment']]

                fig_dash = go.Figure(go.Bar(
                    x=dash_counts['Sentiment'],
                    y=dash_counts['Count'],
                    marker=dict(color=bar_colors),
                    text=dash_counts['Count'],
                    textposition='outside'
                ))
                fig_dash.update_layout(**chart_layout, showlegend=False, xaxis_title="User Verdict", yaxis_title="Respondents")
                st.plotly_chart(fig_dash, width='stretch')  


# ------------ TAB 3: Advanced AI Prediction & Explainability Suite ------------


with tab3:
    st.markdown("### 🧠 Predictive User Intelligence & Adoption Suite")
    st.caption("Real-time Bayesian/Ensemble inference pipeline with model explainability, persona simulation, and decision matrices.")

    try:
        # Load artifacts
        model = joblib.load("models/dashboard_predictor.pkl")
        encoders = joblib.load("models/encoders.pkl")
        target_le = joblib.load("models/target_encoder.pkl")

        # ----------------------------------------------------
        # 1. Quick-Load Persona Profiles (Simulation Presets)
        # ----------------------------------------------------
        preset_cols = st.columns([3, 1])
        with preset_cols[0]:
            persona_preset = st.selectbox(
                "⚡ Load Synthetic Persona Preset (Optional)",
                options=[
                    "Custom Configuration",
                    "College Student (Novice Crypto Trader)",
                    "Salaried Professional (Long-Term Mutual Funds)",
                    "Active Business Trader (High Engagement)"
                ]
            )

        # Dynamic defaults based on preset selection
        defaults = {
            "age": 22, "gender": encoders['Gender'].classes_[0],
            "occ": encoders['Occupation'].classes_[0],
            "inv": encoders['Invested_in_Stock_Market'].classes_[0],
            "pref": encoders['Investment_Interest'].classes_[0],
            "app": encoders['Uses_Stock_Market_Apps_Websites'].classes_[0],
            "vis": encoders['Visualization_Importance'].classes_[0]
        }

        if persona_preset == "College Student (Novice Crypto Trader)":
            defaults.update({"age": 20, "occ": "Student", "inv": "Yes", "pref": "Cryptocurrency"})
        elif persona_preset == "Salaried Professional (Long-Term Mutual Funds)":
            defaults.update({"age": 32, "occ": "Private Employee", "inv": "Yes", "pref": "Mutual Funds"})
        elif persona_preset == "Active Business Trader (High Engagement)":
            defaults.update({"age": 42, "occ": "Business", "inv": "Yes", "pref": "Stocks"})

        # ----------------------------------------------------
        # 2. Interactive Parameter Studio
        # ----------------------------------------------------
        with st.container(border=True):
            st.markdown("#### ⚙️ Feature Engineering & Parameter Studio")
            
            p1, p2, p3 = st.columns(3)
            with p1:
                age_val = st.slider("🎯 Age Demographic", 18, 70, int(defaults["age"]))
                gender_opts = list(encoders['Gender'].classes_)
                gender_val = st.selectbox("⚧ Gender", gender_opts, index=gender_opts.index(defaults["gender"]) if defaults["gender"] in gender_opts else 0)
            
            with p2:
                occ_opts = list(encoders['Occupation'].classes_)
                occ_val = st.selectbox("💼 Professional Segment", occ_opts, index=occ_opts.index(defaults["occ"]) if defaults["occ"] in occ_opts else 0)
                inv_opts = list(encoders['Invested_in_Stock_Market'].classes_)
                inv_val = st.selectbox("📈 Active in Capital Markets?", inv_opts, index=inv_opts.index(defaults["inv"]) if defaults["inv"] in inv_opts else 0)

            with p3:
                pref_opts = list(encoders['Investment_Interest'].classes_)
                pref_val = st.selectbox("💰 Primary Asset Interest", pref_opts, index=pref_opts.index(defaults["pref"]) if defaults["pref"] in pref_opts else 0)
                app_opts = list(encoders['Uses_Stock_Market_Apps_Websites'].classes_)
                app_val = st.selectbox("📱 Utilizes Market Platforms", app_opts, index=app_opts.index(defaults["app"]) if defaults["app"] in app_opts else 0)

            vis_opts = list(encoders['Visualization_Importance'].classes_)
            vis_val = st.select_slider(
                "📊 Valuation of Visual Analytics & Moving Averages",
                options=vis_opts,
                value=defaults["vis"] if defaults["vis"] in vis_opts else vis_opts[0]
            )

            predict_btn = st.button("🚀 Execute Neural / Classifier Inference", type="primary", width='stretch')

        # ----------------------------------------------------
        # 3. Inference Engine & Visual Diagnostics
        # ----------------------------------------------------
        if predict_btn:
            # Transform Inputs into Model Vector
            input_dict = {
                'Gender': encoders['Gender'].transform([gender_val])[0],
                'Age': age_val,
                'Occupation': encoders['Occupation'].transform([occ_val])[0],
                'Invested_in_Stock_Market': encoders['Invested_in_Stock_Market'].transform([inv_val])[0],
                'Investment_Interest': encoders['Investment_Interest'].transform([pref_val])[0],
                'Uses_Stock_Market_Apps_Websites': encoders['Uses_Stock_Market_Apps_Websites'].transform([app_val])[0],
                'Visualization_Importance': encoders['Visualization_Importance'].transform([vis_val])[0]
            }
            input_df = pd.DataFrame([input_dict])

            # Predict Class and Probabilities
            pred_numeric = model.predict(input_df)[0]
            pred_class = str(target_le.inverse_transform([pred_numeric])[0])
            
            has_proba = hasattr(model, "predict_proba")
            class_labels = list(target_le.classes_)
            proba_vector = model.predict_proba(input_df)[0] if has_proba else None

            # Diagnostics HUD
            st.divider()
            hud1, hud2, hud3, hud4 = st.columns(4)
            with hud1:
                st.metric("Primary Adoption State", pred_class)
            with hud2:
                top_confidence = np.max(proba_vector) * 100 if proba_vector is not None else 100.0
                st.metric("Inference Confidence", f"{top_confidence:.1f}%")
            with hud3:
                risk_profile = "Aggressive" if pref_val in ["Cryptocurrency", "Stocks"] else "Conservative"
                st.metric("Risk Profile Tag", risk_profile)
            with hud4:
                engagement_tier = "Tier 1 (High)" if inv_val == "Yes" and app_val == "Yes" else "Tier 2 / Casual"
                st.metric("Engagement Segment", engagement_tier)

            # Detailed Breakdown Cards
            card_col1, card_col2 = st.columns([1.2, 1.8])
            
            with card_col1:
                with st.container(border=True):
                    st.markdown("##### 📊 Class Distribution Probabilities")
                    if proba_vector is not None:
                        proba_df = pd.DataFrame({
                            "State": class_labels,
                            "Probability": proba_vector
                        }).sort_values(by="Probability", ascending=False)
                        st.bar_chart(proba_df.set_index("State"), color="#1f77b4")
                    else:
                        st.info("Predictive model does not expose multi-class probability vectors.")

            with card_col2:
                with st.container(border=True):
                    st.markdown("##### 📌 Strategic Product Recommendations")
                    if pred_class.strip().lower() == "yes":
                        st.success("🎯 **High Lifetime Adoption Persona**")
                        st.markdown("""
                        - Enable **Advanced Technical Indicators** (SMA 50, EMA 200, RSI).
                        - Default to candlestick visualization with real-time refresh hooks.
                        - Offer automated threshold alerts for trend breakouts.
                        """)
                    elif pred_class.strip().lower() == "maybe":
                        st.warning("⚖️ **Hesitant / Exploratory Segment**")
                        st.markdown("""
                        - Present onboarding guided tours for Moving Average interpretation.
                        - Enable simplified high-contrast view options.
                        - Surface summary signals (e.g., 'Bullish Crossover detected') rather than raw numbers.
                        """)
                    else:
                        st.error("📉 **Averse / Non-Adoption Persona**")
                        st.markdown("""
                        - Emphasize passive asset performance summaries (FD / Gold tracking).
                        - Reduce technical charting noise and focus on net worth snapshots.
                        """)

            # ----------------------------------------------------
            # 4. Feature Importance Explainer (if supported)
            # ----------------------------------------------------
            if hasattr(model, "feature_importances_"):
                with st.expander("🔍 Inspect Global Feature Attributions (Explainability)"):
                    feat_imp = pd.Series(model.feature_importances_, index=input_df.columns).sort_values(ascending=True)
                    st.bar_chart(feat_imp)
                    st.caption("Values reflect relative Gini importance computed during decision tree splits.")

    except Exception as e:
        st.error(f"⚠️ Model Pipeline Execution Failure: {e}")


