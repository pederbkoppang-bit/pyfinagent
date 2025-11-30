# pages/1_📈_Portfolio.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import the helper module
try:
    from tools.portfolio import get_live_portfolio_state, execute_cash_operation, get_historical_equity, get_trade_history, MAX_DRAWDOWN, MAX_GROSS_LEVERAGE
except ImportError:
    st.error("Failed to import tools/portfolio.py. Ensure the module is correctly placed.")
    st.stop()

# --- Configuration ---
st.set_page_config(page_title="PyFinAgent Portfolio Management", layout="wide", page_icon="📈")

# Set User Context (Multi-tenancy simulation)
USER_ID = "User_123"
ACCOUNT_ID = "Account_Main_USD"

# --- Helper Functions ---
def create_risk_gauge(current_value, max_value, title, unit=""):
    """Creates a Plotly Indicator Gauge for Risk visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current_value,
        title = {'text': f"{title} (Limit: {max_value:.2f}{unit})", 'font': {'size': 16}},
        # Delta shows distance to the limit; increasing (getting closer) is bad (red)
        delta = {'reference': max_value, 'relative': False, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, max_value*1.1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value*0.7], 'color': 'lightgray'},
                {'range': [max_value*0.7, max_value*0.9], 'color': 'orange'},
                {'range': [max_value*0.9, max_value*1.1], 'color': 'red'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value}}))
    # Optimize layout spacing
    fig.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10))
    return fig

@st.cache_data(ttl=300)
def load_historical_data(user_id, account_id):
    """Cached function to load historical equity data."""
    df = get_historical_equity(user_id, account_id)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp')

@st.cache_data(ttl=300)
def load_trade_history(user_id, account_id):
    """Cached function to load trade history."""
    df = get_trade_history(user_id, account_id)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp', ascending=False)
    return df
# --- Data Loading & Refresh Logic ---
st.title(f"📈 Portfolio Management Dashboard")

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"Account: `{ACCOUNT_ID}`")
    st.caption(f"User ID: `{USER_ID}`")
with col2:
    if st.button("🔄 Refresh Live Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun() # Use st.rerun for a cleaner refresh

with st.spinner("Calculating Live Portfolio State..."):
    portfolio_state = get_live_portfolio_state(USER_ID, ACCOUNT_ID)

if "error" in portfolio_state:
    st.error(portfolio_state["error"])
    st.stop()

df_holdings = portfolio_state["holdings_df"]
metrics = portfolio_state["metrics"]
timestamp = portfolio_state["timestamp"]

equity = metrics.get('total_equity', 0)
cash = metrics.get('cash_balance', 0)
day_pnl = metrics.get('daily_pnl', 0)
leverage = metrics.get('leverage', 0)
drawdown = metrics.get('current_drawdown', 0)

st.caption(f"Live data hybridized at: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

# Load historical data
try:
    equity_curve_df = load_historical_data(USER_ID, ACCOUNT_ID)
except Exception as e:
    st.warning(f"Could not load historical equity curve: {e}")
    equity_curve_df = pd.DataFrame()

# Load trade history data
try:
    trade_history_df = load_trade_history(USER_ID, ACCOUNT_ID)
except Exception as e:
    st.warning(f"Could not load trade history: {e}")
    trade_history_df = pd.DataFrame()
# --- UI Layout: Tab-based Interface ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Overview", "📂 Holdings Analysis", "📜 Trade History", "💵 Cash Management"])

with tab1:
    st.header("Key Metrics")
    hud1, hud2, hud3 = st.columns(3)

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Equity (USD)", value=f"${equity:,.2f}")
            st.caption(f"Cash Balance: ${cash:,.2f}")

        with col2:
            prev_equity = equity - day_pnl
            pnl_pct = f"{(day_pnl/prev_equity)*100:.2f}%" if prev_equity > 0 else "0%"
            st.metric(label="Daily P&L (Snapshot)", value=f"${day_pnl:,.2f}", delta=pnl_pct)

        with col3:
            st.metric(label="Gross Exposure", value=f"${metrics.get('gross_exposure', 0):,.2f}")
            st.caption(f"Net Exposure: ${metrics.get('net_exposure', 0):,.2f}")

    st.header("Risk Utilization")
    with st.container(border=True):
        risk1, risk2 = st.columns(2)
        with risk1:
            fig_lev = create_risk_gauge(leverage, MAX_GROSS_LEVERAGE, "Leverage", "x")
            st.plotly_chart(fig_lev, use_container_width=True)
        with risk2:
            fig_dd = create_risk_gauge(drawdown * 100, MAX_DRAWDOWN * 100, "Drawdown (Live)", "%")
            st.plotly_chart(fig_dd, use_container_width=True)

    st.header("Historical Performance")
    with st.container(border=True):
        if equity_curve_df.empty:
            st.info("Historical performance data is not available.")
        else:
            # Time range selection
            time_ranges = {"1D": 1, "1W": 7, "1M": 30, "6M": 180, "1Y": 365, "5Y": 365*5}
            selected_range = st.radio(
                "Select Time Range",
                options=["1D", "1W", "1M", "6M", "1Y", "5Y", "MAX"],
                index=2, # Default to 1M
                horizontal=True,
            )

            # Filter data based on selection
            end_date = equity_curve_df.index.max()
            if selected_range != "MAX":
                start_date = end_date - timedelta(days=time_ranges[selected_range])
                chart_data = equity_curve_df[equity_curve_df.index >= start_date]
            else:
                chart_data = equity_curve_df

            # Create line chart
            if not chart_data.empty:
                # Determine line color based on performance over the selected period
                start_equity = chart_data['equity'].iloc[0]
                end_equity = chart_data['equity'].iloc[-1]
                line_color = "green" if end_equity >= start_equity else "red"

                fig_perf = px.line(
                    chart_data,
                    x=chart_data.index,
                    y="equity",
                    title=f"Account Equity ({selected_range})",
                    labels={'timestamp': 'Date', 'equity': 'Equity (USD)'}
                )
                fig_perf.update_layout(
                    margin=dict(t=40, b=10, l=10, r=10),
                    xaxis_title=None,
                    yaxis_title="Equity (USD)",
                    showlegend=False
                )
                fig_perf.update_traces(line_color=line_color, line_width=2)
                st.plotly_chart(fig_perf, use_container_width=True)
            else:
                st.info(f"No data available for the selected '{selected_range}' time range.")

with tab2:
    st.header(f"Active Holdings ({len(df_holdings)})")
    if df_holdings.empty:
        st.info("No active positions currently open.")
    else:
        st.dataframe(
            df_holdings,
            use_container_width=True,
            hide_index=True,
            column_order=(
                "ticker", "sector", "quantity", "avg_entry_price", "current_price",
                "market_value", "weight", "pnl_unrealized", "pnl_pct", "sparkline_7d", "strategy_tag"
            ),
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
                "avg_entry_price": st.column_config.NumberColumn("Entry Price", format="$%.4f"),
                "current_price": st.column_config.NumberColumn("Live Price", format="$%.4f"),
                "market_value": st.column_config.NumberColumn("Market Value", format="$%,.2f"),
                "weight": st.column_config.ProgressColumn(
                    "Weight (Net)", format="%.2f%%", min_value=-1.0, max_value=1.0
                ),
                "pnl_unrealized": st.column_config.NumberColumn("P&L (Unr.)", format="$%,.2f"),
                "pnl_pct": st.column_config.NumberColumn("P&L %", format="%.2f%%"),
                "sparkline_7d": st.column_config.LineChartColumn("Price (7d)", width="medium"),
                "strategy_tag": st.column_config.TextColumn("Strategy")
            }
        )

    if not df_holdings.empty:
        with st.expander("Visualise Portfolio Allocation", expanded=True):
            chart_df = df_holdings.copy()
            chart_df['abs_exposure'] = chart_df['market_value'].abs()
            chart_df['sector'] = chart_df['sector'].fillna('Other')
            sector_allocation = chart_df.groupby('sector')['abs_exposure'].sum().reset_index()

            st.subheader("Sector Exposure (Gross)")
            fig_sector = px.pie(
                sector_allocation, values='abs_exposure', names='sector', hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_sector.update_traces(textposition='inside', textinfo='percent+label')
            fig_sector.update_layout(showlegend=False, margin=dict(t=10, b=10, l=0, r=0))
            st.plotly_chart(fig_sector, use_container_width=True)

with tab3:
    st.header(f"Trade Execution History ({len(trade_history_df)})")
    if trade_history_df.empty:
        st.info("No trade history found for this account.")
    else:
        st.dataframe(
            trade_history_df,
            use_container_width=True,
            hide_index=True,
            column_order=(
                "timestamp", "ticker", "side", "quantity", "price", "cost_basis", "strategy_tag"
            ),
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Execution Time", format="YYYY-MM-DD HH:mm:ss"),
                "ticker": st.column_config.TextColumn("Ticker"),
                "side": st.column_config.TextColumn("Side"),
                "quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
                "price": st.column_config.NumberColumn("Execution Price", format="$%.4f"),
                "cost_basis": st.column_config.NumberColumn("Cost Basis", format="$%,.2f"),
                "strategy_tag": st.column_config.TextColumn("Strategy")
            }
        )


with tab4:
    st.header("Ledger Management")
    with st.container(border=True):
        with st.form(key="cash_ops_form"):
            st.subheader("Execute Cash Operation")
            col_op, col_amt, col_curr = st.columns(3)

            with col_op:
                operation = st.selectbox("Operation Type", ["DEPOSIT", "WITHDRAWAL"])
            with col_amt:
                amount = st.number_input("Amount", min_value=0.01, step=100.0)
            with col_curr:
                currency_input = st.selectbox("Currency", ["USD"])

            submitted = st.form_submit_button("✅ Execute Transaction", use_container_width=True)

            if submitted:
                if operation == "WITHDRAWAL" and currency_input == "USD" and amount > cash:
                     st.error(f"Insufficient USD funds. Available cash: ${cash:,.2f}")
                else:
                    with st.spinner("Logging transaction to ledger..."):
                        success = execute_cash_operation(USER_ID, ACCOUNT_ID, amount, operation, currency_input)
                        if success:
                            st.success(f"Transaction logged successfully.")
                            st.rerun()
                        else:
                            st.error("Transaction failed. Check logs.")