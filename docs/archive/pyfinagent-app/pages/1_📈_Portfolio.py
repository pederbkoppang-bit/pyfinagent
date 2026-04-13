# pages/1_📈_Portfolio.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import streamlit.components.v1 as components

# Import the helper module
try:
    from tools.portfolio import get_live_portfolio_state, execute_cash_operation, get_historical_equity, get_trade_history, fetch_benchmark_data, MAX_DRAWDOWN, MAX_GROSS_LEVERAGE
except ImportError:
    st.error("Failed to import tools/portfolio.py. Ensure the module is correctly placed.")
    st.stop()

# --- Configuration ---
st.set_page_config(page_title="PyFinAgent Portfolio Management", layout="wide", page_icon="📈")

# Set User Context (Multi-tenancy simulation)
USER_ID = "User_123"

@st.cache_data(ttl=300)
def load_historical_data(user_id, account_id):
    """Cached function to load historical equity data."""
    df = get_historical_equity(user_id, account_id)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp')

# --- Mock Data for Multi-Account Demo ---
@st.cache_data
def get_mock_historical_equity_eur():
    """Generates mock historical equity for a second account."""
    # Use a relative date range to ensure data is always generated, avoiding timezone/start-of-day issues.
    end_date = datetime.now()
    date_rng = pd.date_range(start=end_date - relativedelta(years=2), end=end_date, freq='D')
    np.random.seed(21)
    returns = np.random.normal(loc=0.0003, scale=0.02, size=len(date_rng))
    equity = 50000 * (1 + returns).cumprod()
    df = pd.DataFrame({'timestamp': date_rng, 'equity': equity})
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

# --- Charting Functions (New) ---
def create_hero_equity_chart(df, total_equity, total_pnl, benchmark_df=None):
    """Creates a pro-level equity chart with optional benchmark comparison."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # Configure range selector buttons
    end_date = df.index.max()

    # --- Dynamic Percentage Change Traces ---
    # Pre-calculate percentage change for each time range to avoid JS.
    # The button will make the corresponding trace visible.
    time_ranges = {
        "1D": end_date - timedelta(days=1),
        "1W": end_date - timedelta(days=7),
        "1M": end_date - relativedelta(months=1),
        "6M": end_date - relativedelta(months=6),
        "YTD": df.index[df.index.year == end_date.year].min(),
        "1Y": end_date - relativedelta(years=1),
        "MAX": df.index.min()
    }

    trace_visibility = [False] * len(time_ranges)
    trace_visibility[3] = True # Default to 6M

    all_annotations = []
    all_yaxis_ranges = []

    # Add the main equity line trace first, but it will be controlled by buttons
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['equity'],
        # Initially hide it; buttons will set the correct data and visibility
        visible=False
    ))


    for i, (range_label, start_date) in enumerate(time_ranges.items()):
        # Filter the DataFrame for the specific time range
        range_df = df[df.index >= start_date]
        yaxis_range = [None, None] # Default empty range
        range_pnl_text = ""
        if not range_df.empty:
            # Determine color based on the performance *within this specific range*
            range_color = "green" if range_df['equity'].iloc[-1] >= range_df['equity'].iloc[0] else "red"
            first_val = range_df['equity'].iloc[0]
            pct_change = (range_df['equity'] / first_val) - 1

            # Calculate y-axis range with padding for better readability
            min_equity = range_df['equity'].min()
            max_equity = range_df['equity'].max()
            # Reserve top 25% of the view for annotations to prevent overlap
            data_range = max_equity - min_equity
            padding = data_range / 3 # This makes the data range occupy ~75% of the axis
            yaxis_range = [min_equity, max_equity + padding]

            # Calculate P&L for the annotation
            pnl_value = range_df['equity'].iloc[-1] - first_val
            pnl_pct = (pnl_value / first_val) * 100 if first_val != 0 else 0
            pnl_sign = "+" if pnl_value >= 0 else ""
            pnl_color = "#00FF00" if pnl_value >= 0 else "#FF0000"
            range_pnl_text = f"<span style='color:{pnl_color};'>{pnl_sign}${abs(pnl_value):,.2f} ({pnl_sign}{abs(pnl_pct):.2f}%)</span> {range_label}"
        else:
            range_color = "green" # Default color for empty ranges
            pct_change = pd.Series() # Create an empty series if no data
            range_pnl_text = f"No data for {range_label}"
        all_yaxis_ranges.append(yaxis_range)

        # Create a list of annotations for this specific time range
        current_annotations = [
            dict(text="Holdings", align='left', showarrow=False, xref='paper', yref='paper', x=0.02, y=0.98, font=dict(size=16, color="#a0a0a0")),
            dict(text=f"<b>${total_equity:,.2f}</b>", align='left', showarrow=False, xref='paper', yref='paper', x=0.02, y=0.88, font=dict(size=32, color="#FFFFFF")),
            dict(text=range_pnl_text, align='left', showarrow=False, xref='paper', yref='paper', x=0.02, y=0.78, font=dict(size=16, color="#a0a0a0"))
        ]
        all_annotations.append(current_annotations)

        # Add a separate, correctly-scoped equity trace for each button
        fig.add_trace(go.Scatter(
            x=range_df.index,
            y=range_df['equity'],
            mode='lines',
            line=dict(color=range_color, width=2.5),
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.05)' if range_color == 'green' else 'rgba(255, 0, 0, 0.05)',
            name='Equity',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Equity: $%{y:,.2f}<extra></extra>',
            yaxis='y1',
            visible=trace_visibility[i] # Set visibility based on default
        ))

        # Add benchmark traces for this time range
        if benchmark_df is not None and not benchmark_df.empty:
            for ticker in benchmark_df.columns:
                benchmark_range_df = benchmark_df.loc[benchmark_df.index >= start_date]
                if not benchmark_range_df.empty:
                    first_benchmark_val = benchmark_range_df[ticker].iloc[0]
                    benchmark_pct_change = (benchmark_range_df[ticker] / first_benchmark_val) - 1
                    fig.add_trace(go.Scatter(
                        x=benchmark_range_df.index, y=benchmark_pct_change,
                        mode='lines', line=dict(width=1.5, dash='dash'),
                        name=ticker, hovertemplate=f'{ticker}: %{{y:.2%}}<extra></extra>',
                        visible=trace_visibility[i], yaxis='y2'
                    ))

        fig.add_trace(go.Scatter(
            x=range_df.index, # Use the filtered index
            y=pct_change,
            mode='lines',
            line=dict(color='rgba(173, 216, 230, 0.4)', width=1, dash='dot'),
            name=f'Perf {range_label}',
            hovertemplate='Performance: %{y:.2%}<extra></extra>',
            visible=trace_visibility[i],
            yaxis='y2'
        ))

    # Create visibility arrays for the buttons
    buttons = []
    num_benchmarks = len(benchmark_df.columns) if benchmark_df is not None else 0
    num_traces_per_range = 2 + num_benchmarks # (equity, performance, ...benchmarks)
    total_traces = 1 + len(time_ranges) * num_traces_per_range # 1 (hidden base) + N * 2
    for i in range(len(time_ranges)):
        visibility = [False] * total_traces
        for j in range(num_traces_per_range):
            trace_index = 1 + (i * num_traces_per_range) + j
            visibility[trace_index] = True
        buttons.append(visibility)

    # --- Calculate Initial Y-Axis Range ---
    # Set the default visible range to 6 months to calculate the initial y-axis range
    six_months_ago = end_date - relativedelta(months=6)
    if six_months_ago < df.index.min():
        six_months_ago = df.index.min()

    # Calculate initial y-axis range for the default view (6M)
    initial_yaxis_range = all_yaxis_ranges[3]


    # Set initial annotations for the default view (6M)
    fig.layout.annotations = all_annotations[3]

    fig.update_layout(
        xaxis=dict(
            type='date',
            showgrid=True, # Enable X-axis grid
            gridcolor='rgba(255, 255, 255, 0.05)', # Universal light grid color
            visible=True,
            tickfont=dict(color='#a0a0a0'), # Show X-axis tick labels
            rangeslider=dict(visible=False),
            range=[six_months_ago, end_date] # Set initial range here
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="up",
            active=3,  # Default to 6M
            showactive=True, # Makes the button dynamic
            buttons=[dict(
                label=label,
                method="update",
                args=[
                    {"visible": buttons[i]},
                    {
                        "xaxis.range": [time_ranges[label], end_date],
                        "yaxis.range": all_yaxis_ranges[i],
                        "annotations": all_annotations[i]
                    }
                ]) for i, label in enumerate(time_ranges.keys())
            ],
            x=0.99,
            xanchor="right",
            y=0.01,
            yanchor="bottom",
            bgcolor='rgba(38, 38, 38, 0.9)',
            font=dict(color='#a0a0a0'),
            bordercolor='rgba(0,0,0,0)',
        )],
        height=450,
        margin=dict(l=60, r=40, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis=dict(
            visible=True, # Show primary Y-axis
            title="",
            autorange=False, # Disable autorange to use our calculated range
            range=initial_yaxis_range, # Set initial range
            showgrid=True, # Enable Y-axis grid
            gridcolor='rgba(255, 255, 255, 0.05)', # Universal light grid color
            tickformat="~s", # Use compact number format (e.g., 1.2M, 5k)
            tickfont=dict(color='#a0a0a0'),
        ),
        yaxis2=dict(
            title="", # Remove title
            overlaying='y',
            side='right',
            tickformat='.0%', # Format as percentage
            autorange=True,
            fixedrange=False,
            showticklabels=True, # Show the tick labels (e.g., 10%, 20%)
            ticksuffix=" ", # Add a space to prevent overlap with chart edge
        ),
        showlegend=True, # Show the legend for benchmarks
        font_color='#a0a0a0',
        hovermode='x unified',
        dragmode='pan'
    )

    return fig

# --- Data Loading & Refresh Logic ---
st.title(f"📈 Portfolio Management Dashboard")

# --- Account Selection Header ---
ACCOUNTS = {"Account_Main_USD": "Main (USD)", "Account_Margin_EUR": "Margin (EUR)"}
selected_accounts = st.multiselect(
    "Select Accounts",
    options=list(ACCOUNTS.keys()),
    format_func=lambda x: ACCOUNTS[x],
    default=list(ACCOUNTS.keys())
)

if not selected_accounts:
    st.warning("Please select at least one account to view.")
    st.stop()

# --- Data Aggregation for Selected Accounts ---
all_equity_curves = []
total_equity, total_cash, total_day_pnl = 0, 0, 0

if "Account_Main_USD" in selected_accounts:
    portfolio_state = get_live_portfolio_state(USER_ID, "Account_Main_USD")
    equity_curve_df = load_historical_data(USER_ID, "Account_Main_USD")
    all_equity_curves.append(equity_curve_df)
    total_equity += portfolio_state["metrics"].get('total_equity', 0)
    total_cash += portfolio_state["metrics"].get('cash_balance', 0)
    total_day_pnl += portfolio_state["metrics"].get('daily_pnl', 0)

if "Account_Margin_EUR" in selected_accounts:
    # Using mock data for the second account
    eur_equity_curve = get_mock_historical_equity_eur()
    all_equity_curves.append(eur_equity_curve)
    # Mock live metrics for EUR account
    total_equity += eur_equity_curve['equity'].iloc[-1]
    total_cash += 25000 # Mock cash
    total_day_pnl -= 550.75 # Mock PnL

# Combine historical data
if all_equity_curves:
    combined_equity_df = pd.concat(all_equity_curves).groupby('timestamp').sum().sort_index()
else:
    combined_equity_df = pd.DataFrame()


# --- UI Layout: Tab-based Interface ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Overview", "📂 Holdings Analysis", "📜 Trade History", "💵 Cash Management"])

# --- TAB 1: PORTFOLIO OVERVIEW (REDESIGNED) ---
with tab1:
    # --- Benchmark Selection (Moved here) ---
    benchmark_tickers = st.multiselect(
        "Compare Portfolio Performance",
        options=['SPY', 'QQQ', 'VT', 'BTC-USD', 'GLD'],
        default=[],
        help="Select benchmarks to compare against your portfolio's percentage performance."
    )
    # Fetch benchmark data based on selection
    benchmark_df = fetch_benchmark_data(benchmark_tickers)

    # --- Hero Equity Chart ---
    hero_chart = create_hero_equity_chart(combined_equity_df, total_equity, total_day_pnl, benchmark_df)
    st.plotly_chart(hero_chart, use_container_width=True, config={'displayModeBar': False})

with tab2:
    # This tab now needs to be adapted for multi-account selection if desired
    # For now, it will only show data for the first selected account
    if not selected_accounts:
        st.info("Select an account to see holdings.")
        st.stop()

    with st.spinner("Calculating Live Portfolio State..."):
        # We'll display data for the first selected account for simplicity in this tab
        active_account_id = selected_accounts[0]
        portfolio_state = get_live_portfolio_state(USER_ID, active_account_id)

    if "error" in portfolio_state:
        st.error(portfolio_state["error"])
        st.stop()

    df_holdings = portfolio_state["holdings_df"]

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

    # --- Advanced Visualizations (HTML/JS Injection) ---
    if not df_holdings.empty:
        # Convert dataframe to JSON for JS components
        df_holdings['abs_market_value'] = df_holdings['market_value'].abs()
        holdings_json = df_holdings.to_json(orient='records')
        
        # Determine the top holding for the default TradingView chart
        top_holding_ticker = df_holdings.iloc[0]['ticker']

        # --- 1. Portfolio Treemap (Plotly.js) ---
        st.subheader("Portfolio Allocation Treemap")
        with st.container(border=True):
            treemap_html = f"""
            <div id="treemap" style="width:100%; height:500px;"></div>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                const holdings = {holdings_json};
                
                const labels = holdings.map(h => h.ticker);
                const parents = holdings.map(h => h.sector || 'Other');
                const values = holdings.map(h => h.abs_market_value);
                const pnl_pct = holdings.map(h => h.pnl_pct);

                const customdata = holdings.map(h => `
                    <b>Sector:</b> ${{h.sector || 'N/A'}}<br>
                    <b>Market Value:</b> $${{h.market_value.toLocaleString('en-US', {{maximumFractionDigits: 2}})}}<br>
                    <b>P&L %:</b> ${{(h.pnl_pct * 100).toFixed(2)}}%`
                `);

                const data = [{{
                    type: "treemap",
                    labels: labels,
                    parents: parents,
                    values: values,
                    customdata: customdata,
                    hovertemplate: '<b>%{label}</b><br>%{{customdata}}<extra></extra>',
                    marker: {{
                        colors: pnl_pct,
                        colorscale: [
                            ['0.0', 'rgb(211, 63, 63)'], // Red for negative
                            ['0.5', 'rgb(128, 128, 128)'], // Gray for zero
                            ['1.0', 'rgb(63, 211, 63)']  // Green for positive
                        ],
                        cmid: 0,
                        cmin: -0.10, // Set a reasonable min/max for color scaling
                        cmax: 0.10,
                    }},
                    textinfo: "label+value"
                }}];

                const layout = {{
                    margin: {{l: 0, r: 0, b: 0, t: 0}},
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#FFFFFF' }}
                }};

                Plotly.newPlot('treemap', data, layout, {{responsive: true}});
            </script>
            """
            components.html(treemap_html, height=510)

        # --- 2. Holdings Heatmap (Vanilla JS) ---
        st.subheader("Holdings Heatmap (Live P&L %)")
        with st.container(border=True):
            heatmap_html = f"""
            <style>
                :root {{
                    --heatmap-bg: #1a1a1a;
                    --border-color: #444;
                    --text-color: #e1e1e1;
                    --green-glow: rgba(63, 211, 63, 0.7);
                    --red-glow: rgba(211, 63, 63, 0.7);
                }}
                .heatmap-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                    gap: 10px;
                    padding: 10px;
                }}
                .heatmap-tile {{
                    background-color: var(--heatmap-bg);
                    border: 1px solid var(--border-color);
                    border-radius: 5px;
                    padding: 15px 10px;
                    text-align: center;
                    color: var(--text-color);
                    font-family: monospace;
                }}
                .tile-ticker {{ font-size: 1.2em; font-weight: bold; }}
                .tile-pnl {{ font-size: 1.0em; margin-top: 5px; }}
            </style>
            <div id="heatmap-container" class="heatmap-grid"></div>
            <script>
                const holdingsData = {holdings_json};
                const container = document.getElementById('heatmap-container');

                // Function to map P&L % to a color
                function getPnlColor(pnl) {{
                    const intensity = Math.min(Math.abs(pnl) * 5, 1); // Scale intensity, cap at 1
                    if (pnl > 0) {{
                        return `rgba(63, 211, 63, ${intensity})`; // Green with variable alpha
                    }} else if (pnl < 0) {{
                        return `rgba(211, 63, 63, ${intensity})`; // Red with variable alpha
                    }}
                    return 'var(--heatmap-bg)'; // Neutral
                }}

                holdingsData.forEach(h => {{
                    const tile = document.createElement('div');
                    tile.className = 'heatmap-tile';
                    tile.style.backgroundColor = getPnlColor(h.pnl_pct);

                    const tickerDiv = document.createElement('div');
                    tickerDiv.className = 'tile-ticker';
                    tickerDiv.textContent = h.ticker;

                    const pnlDiv = document.createElement('div');
                    pnlDiv.className = 'tile-pnl';
                    pnlDiv.textContent = `${(h.pnl_pct * 100).toFixed(2)}%`;

                    tile.appendChild(tickerDiv);
                    tile.appendChild(pnlDiv);
                    container.appendChild(tile);
                }});
            </script>
            """
            components.html(heatmap_html, height=300, scrolling=True)

        # --- 3. TradingView Widget ---
        st.subheader("Live Chart")
        with st.container(border=True):
            tradingview_html = f"""
            <!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container" style="height:100%;width:100%">
              <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget(
              {{
              "autosize": true,
              "symbol": "NASDAQ:{top_holding_ticker}",
              "interval": "D",
              "timezone": "Etc/UTC",
              "theme": "dark",
              "style": "1",
              "locale": "en",
              "enable_publishing": false,
              "allow_symbol_change": true,
              "container_id": "tradingview_chart"
            }}
              );
              </script>
            </div>
            <!-- TradingView Widget END -->
            """
            components.html(tradingview_html, height=500)

with tab3:
    # For simplicity, this tab will show combined history if multiple accounts are selected
    trade_history_dfs = []
    for acc_id in selected_accounts:
        trade_history_dfs.append(load_trade_history(USER_ID, acc_id))
    trade_history_df = pd.concat(trade_history_dfs).sort_values('timestamp', ascending=False)

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
    # For simplicity, cash operations will target the first selected account
    if not selected_accounts:
        st.info("Select an account to manage cash.")
        st.stop()
    cash_op_account_id = selected_accounts[0]
    cash = get_live_portfolio_state(USER_ID, cash_op_account_id)["metrics"].get('cash_balance', 0)
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
                        success = execute_cash_operation(USER_ID, cash_op_account_id, amount, operation, currency_input)
                        if success:
                            st.success(f"Transaction logged successfully.")
                            st.rerun()
                        else:
                            st.error("Transaction failed. Check logs.")