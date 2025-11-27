import streamlit as st
import pandas as pd
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_price_chart(ticker: str, analysis_dates=None):
    """
    Renders a placeholder chart initially, and then fills it with historical
    stock price data once available from st.session_state.
    Optionally plots markers for past analysis dates.

    Args:
        ticker (str): The stock ticker symbol (e.g., "NVDA") for the chart title.
        analysis_dates (list, optional): A list of dates to mark on the chart.
    """
    # Check if the necessary data is available in the session state report
    quant_data = st.session_state.get('report', {}).get('part_1_5_quant')
    # Corrected path: The historical prices are nested inside the 'part_5_valuation' dictionary.
    historical_prices_json = quant_data.get('part_5_valuation', {}).get('historical_prices') if quant_data else None

    if not historical_prices_json:
        # If there's no data, simply don't display the chart.
        # A placeholder can be added here if desired for the "loading" state.
        st.info("Historical price data not available for chart.")
        return

    # --- Data Processing ---
    try:
        price_df = pd.read_json(historical_prices_json, orient='split')
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df.set_index('Date', inplace=True)
    except (ValueError, KeyError) as e:
        logging.error(f"Could not parse historical price data: {e}")
        st.warning("Could not display price chart. The data format may be incorrect.")
        return

    st.subheader("Historical Performance")

    # --- UI Controls for the Chart ---
    st.write("#### Chart Options")
    cols = st.columns(4)
    with cols[0]:
        show_sma_50 = st.checkbox("Show SMA 1", value=True)
    with cols[1]:
        show_sma_200 = st.checkbox("Show SMA 2", value=False)
    with cols[2]:
        show_bb = st.checkbox("Bollinger Bands", value=False)
    with cols[3]:
        show_rsi = st.checkbox("Show RSI", value=True)

    # --- Expander for Customizing Indicator Parameters ---
    with st.expander("Customize Indicator Parameters"):
        param_cols = st.columns(4)
        with param_cols[0]:
            sma_50_period = st.number_input("SMA 1 Period", min_value=1, max_value=200, value=50, step=1)
            sma_200_period = st.number_input("SMA 2 Period", min_value=1, max_value=400, value=200, step=1)
        with param_cols[1]:
            bb_window = st.number_input("BB Window", min_value=1, max_value=100, value=20, step=1)
            bb_std = st.number_input("BB Std. Dev.", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
        with param_cols[2]:
            rsi_period = st.number_input("RSI Period", min_value=1, max_value=100, value=14, step=1)


    # --- Calculate Technical Indicators ---
    price_df[f'SMA_{sma_50_period}'] = price_df['Close'].rolling(window=sma_50_period).mean()
    price_df[f'SMA_{sma_200_period}'] = price_df['Close'].rolling(window=sma_200_period).mean()
    
    # Calculate Bollinger Bands
    price_df[f'SMA_{bb_window}'] = price_df['Close'].rolling(window=bb_window).mean()
    price_df['BB_std'] = price_df['Close'].rolling(window=bb_window).std()
    price_df['BB_Upper'] = price_df[f'SMA_{bb_window}'] + (price_df['BB_std'] * bb_std)
    price_df['BB_Lower'] = price_df[f'SMA_{bb_window}'] - (price_df['BB_std'] * bb_std)

    # Calculate RSI
    delta = price_df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    price_df['RSI'] = 100 - (100 / (1 + rs))

    # --- Create the Plotly Figure ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=('', 'Volume', 'RSI'),
                        row_heights=[0.6, 0.2, 0.2])

    # --- Candlestick Trace ---
    fig.add_trace(go.Candlestick(x=price_df.index,
                                 open=price_df['Open'],
                                 high=price_df['High'],
                                 low=price_df['Low'],
                                 close=price_df['Close'],
                                 name='OHLC'),
                  row=1, col=1)

    # --- Moving Average Traces (conditional) ---
    if show_sma_50:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[f'SMA_{sma_50_period}'],
                                 mode='lines', name=f'{sma_50_period}-Day SMA',
                                 line=dict(color='orange', width=1)),
                      row=1, col=1)
    if show_sma_200:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[f'SMA_{sma_200_period}'],
                                 mode='lines', name=f'{sma_200_period}-Day SMA',
                                 line=dict(color='purple', width=1)),
                      row=1, col=1)
    
    if show_bb:
        # Upper Band
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df['BB_Upper'], mode='lines',
                                 line=dict(color='gray', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
        # Lower Band - use fill to create a shaded area
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df['BB_Lower'], mode='lines',
                                 line=dict(color='gray', width=1, dash='dash'), name='BB Lower',
                                 fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        # Middle Band
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[f'SMA_{bb_window}'], mode='lines',
                                 line=dict(color='gray', width=1), name='BB Middle'), row=1, col=1)

    # --- Volume Bar Trace ---
    # Color bars based on price movement
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in price_df.iterrows()]
    fig.add_trace(go.Bar(x=price_df.index, y=price_df['Volume'],
                         marker_color=colors, name='Volume'),
                  row=2, col=1)
    
    # --- RSI Trace (conditional) ---
    if show_rsi:
        fig.add_trace(go.Scatter(x=price_df.index, y=price_df['RSI'], mode='lines', name='RSI', line=dict(color='blue', width=1)), row=3, col=1)
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1)

    # --- Add Markers for Past Analyses ---
    if analysis_dates:
        for date in analysis_dates:
            fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="blue",
                          annotation_text="Analysis", annotation_position="top left",
                          row=1, col=1)


    # --- Layout and Theming ---
    fig.update_layout(
        title_text=f"{ticker} Stock Price",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False, # Hide the default range slider
        yaxis_title='Price (USD)',
        template='plotly_white', # Use a clean, professional theme
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    # --- Date Range Buttons ---
    fig.update_xaxes(
        row=1, col=1,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig, use_container_width=True)