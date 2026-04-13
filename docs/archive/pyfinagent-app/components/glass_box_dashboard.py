import streamlit as st
import json

def display_glass_box_dashboard(payload: dict):
    """
    Renders the high-fidelity "Glass Box" dashboard using a self-contained
    HTML component with embedded Tailwind CSS and JavaScript.

    Args:
        payload (dict): The hybridized data payload for the frontend.
    """

    # Convert the Python dict to a JSON string to safely inject into the HTML.
    json_payload = json.dumps(payload)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Glass Box Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
        <script src="https://unpkg.com/lucide@latest"></script>

        <style>
            body {{
                background-color: #020617; /* Deep Navy */
                font-family: 'Inter', sans-serif;
                color: #e2e8f0; /* Slate 200 */
            }}
            .font-mono {{
                font-family: 'JetBrains Mono', monospace;
            }}
            .bento-card {{
                background-color: rgba(15, 23, 42, 0.7); /* Slate 900 with transparency */
                border: 1px solid #1e293b; /* Slate 800 */
                border-radius: 1rem;
                padding: 1.5rem;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }}
            .alpha-score-glow {{
                box-shadow: 0 0 30px rgba(56, 189, 248, 0.4), 0 0 60px rgba(56, 189, 248, 0.2);
            }}
            .risk-dot {{
                position: absolute;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: white;
                box-shadow: 0 0 8px white, 0 0 12px white;
                animation: pulse 1.5s infinite ease-in-out;
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(0.8); opacity: 0.7; }}
                50% {{ transform: scale(1.2); opacity: 1; }}
            }}
            .logic-token {{
                display: inline-block;
                background-color: #334155; /* Slate 700 */
                color: #cbd5e1; /* Slate 300 */
                border-radius: 9999px;
                padding: 0.25rem 0.75rem;
                font-size: 0.75rem;
                font-family: 'JetBrains Mono', monospace;
                margin-right: 0.5rem;
                margin-top: 0.5rem;
            }}
            /* Custom scrollbar for chat */
            .chat-messages::-webkit-scrollbar {{
                width: 6px;
            }}
            .chat-messages::-webkit-scrollbar-track {{
                background: transparent;
            }}
            .chat-messages::-webkit-scrollbar-thumb {{
                background: #334155; /* Slate 700 */
                border-radius: 3px;
            }}
        </style>
    </head>
    <body class="p-4 md:p-6">
        <div id="dashboard" class="grid grid-cols-12 gap-6">
            <!-- Main Content Area -->
            <div class="col-span-12 lg:col-span-9">
                <div class="grid grid-cols-12 gap-6">
                    <!-- Alpha Score -->
                    <div id="alpha-score" class="col-span-12 md:col-span-4 bento-card flex flex-col justify-between alpha-score-glow"></div>

                    <!-- Investment Thesis -->
                    <div id="investment-thesis" class="col-span-12 md:col-span-8 bento-card"></div>

                    <!-- Valuation Football Field -->
                    <div id="valuation-field" class="col-span-12 md:col-span-8 bento-card"></div>

                    <!-- Risk Matrix -->
                    <div id="risk-matrix" class="col-span-12 md:col-span-4 bento-card flex flex-col items-center justify-center"></div>
                </div>
            </div>

            <!-- Sidebar / Research Investigator -->
            <div class="col-span-12 lg:col-span-3 bento-card flex flex-col h-[950px]">
                <div class="flex items-center mb-4">
                    <i data-lucide="search" class="w-5 h-5 mr-2 text-sky-400"></i>
                    <h3 class="text-lg font-semibold text-slate-200">Research Investigator</h3>
                </div>
                <div class="flex-grow flex flex-col bg-slate-900/50 rounded-lg p-2">
                    <div class="chat-messages flex-grow overflow-y-auto pr-2">
                        <div class="p-2 text-sm text-slate-400">Welcome. Ask a follow-up question about the analysis. (UI Mockup)</div>
                    </div>
                    <div class="mt-2 flex items-center border border-slate-700 rounded-lg p-1">
                        <input type="text" placeholder="e.g., 'Explain the P/E ratio...'" class="flex-grow bg-transparent focus:outline-none px-2 text-sm">
                        <button class="p-2 rounded-md bg-sky-500 hover:bg-sky-600 transition-colors">
                            <i data-lucide="send" class="w-4 h-4 text-white"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const data = {json_payload};

            // Utility to format large numbers
            function formatNumber(num) {{
                if (num === null || num === undefined) return 'N/A';
                if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
                if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
                if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
                return num.toLocaleString();
            }}

            // --- 1. Alpha Score Tile ---
            function renderAlphaScore() {{
                const score = data.ai_score || 0;
                const score_pct = (score / 10) * 100;
                const recommendation = data.recommendation || {{}};
                
                let colorClass = 'text-slate-400';
                if (recommendation.action?.toLowerCase().includes('buy')) colorClass = 'text-emerald-400';
                if (recommendation.action?.toLowerCase().includes('sell')) colorClass = 'text-rose-400';

                const container = document.getElementById('alpha-score');
                container.innerHTML = `
                    <div>
                        <div class="flex items-center text-slate-400">
                            <i data-lucide="brain-circuit" class="w-5 h-5 mr-2"></i>
                            <h3 class="text-lg font-semibold">Alpha Score</h3>
                        </div>
                        <p class="text-8xl font-bold text-sky-300 font-mono mt-4">${{score.toFixed(2)}}</p>
                    </div>
                    <div>
                        <p class="text-xl font-semibold ${{colorClass}} mb-2">${{recommendation.action || 'Neutral'}}</p>
                        <div class="w-full bg-slate-700 rounded-full h-2.5">
                            <div class="bg-gradient-to-r from-sky-500 to-cyan-400 h-2.5 rounded-full" style="width: ${{score_pct}}%"></div>
                        </div>
                    </div>
                `;
            }}

            // --- 2. Investment Thesis Tile ---
            function renderInvestmentThesis() {{
                const financials = data.financials || {{}};
                const summary = data.summary || "No summary provided.";
                
                // Simple logic token extraction (example)
                const logicTokens = (summary.match(/Logic: [^\\.]+/g) || []).map(t => t.replace('Logic: ', ''));
                if (logicTokens.length === 0) {{
                    logicTokens.push("Holistic Analysis"); // Default token
                }}

                const container = document.getElementById('investment-thesis');
                container.innerHTML = `
                    <div class="flex items-center text-slate-400 mb-4">
                        <i data-lucide="microscope" class="w-5 h-5 mr-2"></i>
                        <h3 class="text-lg font-semibold">Investment Thesis</h3>
                    </div> 
                    <p class="text-slate-300 text-sm leading-relaxed mb-6">${{summary.replace(/Logic: [^\\.]+/g, '')}}</p>
                    <div class="mb-6">
                        ${{logicTokens.map(token => `<span class="logic-token">${{token}}</span>`).join('')}}
                    </div>
                    <div class="grid grid-cols-3 gap-4 border-t border-slate-800 pt-4 mt-auto">
                        <div class="text-center">
                            <p class="text-xs text-slate-400">Revenue</p>
                            <p class="text-xl font-semibold font-mono text-emerald-400">${{formatNumber(financials.revenue)}}</p>
                        </div>
                        <div class="text-center">
                            <p class="text-xs text-slate-400">Net Income</p>
                            <p class="text-xl font-semibold font-mono text-emerald-400">${{formatNumber(financials.net_income)}}</p>
                        </div>
                        <div class="text-center">
                            <p class="text-xs text-slate-400">Market Cap</p>
                            <p class="text-xl font-semibold font-mono text-sky-400">${{formatNumber(financials.market_cap)}}</p>
                        </div>
                    </div>
                `;
            }}

            // --- 3. Valuation Football Field ---
            function renderValuationField() {{
                const container = document.getElementById('valuation-field');
                const ranges = data.valuation_ranges || {{}};
                const currentPrice = data.current_price || 0;

                // Mock data if real data is missing
                const valuationData = [
                    {{ label: 'P/E Ratio Range', low: ranges.pe_ratio ? ranges.pe_ratio * 0.8 : 80, high: ranges.pe_ratio ? ranges.pe_ratio * 1.2 : 120 }},
                    {{ label: 'DCF Analysis', low: 95, high: 135 }},
                    {{ label: 'Analyst Targets', low: 110, high: 150 }},
                    {{ label: 'P/S Ratio Range', low: ranges.ps_ratio ? ranges.ps_ratio * 2.5 : 70, high: ranges.ps_ratio ? ranges.ps_ratio * 3.5 : 98 }},
                ];

                const allValues = [currentPrice, ...valuationData.flatMap(d => [d.low, d.high])];
                const minVal = Math.min(...allValues) * 0.9;
                const maxVal = Math.max(...allValues) * 1.1;

                const chartWidth = 500; // SVG width

                function scaleValue(value) {{
                    return ((value - minVal) / (maxVal - minVal)) * chartWidth;
                }}

                const bars = valuationData.map((d, i) => {{
                    const x = scaleValue(d.low);
                    const width = scaleValue(d.high) - x;
                    const y = i * 40 + 20;
                    return `
                        <text x="0" y="${{y + 8}}" fill="#94a3b8" font-size="12">${{d.label}}</text>
                        <rect x="${{x}}" y="${{y}}" width="${{width}}" height="15" fill="#1e40af" rx="3"></rect>
                        <text x="${{x - 5}}" y="${{y + 12}}" fill="#e2e8f0" font-size="10" text-anchor="end" class="font-mono">${{d.low.toFixed(2)}}</text>
                        <text x="${{x + width + 5}}" y="${{y + 12}}" fill="#e2e8f0" font-size="10" text-anchor="start" class="font-mono">${{d.high.toFixed(2)}}</text>
                    `;
                }}).join('');

                const priceLineX = scaleValue(currentPrice);

                container.innerHTML = `
                    <div class="flex items-center text-slate-400 mb-4">
                        <i data-lucide="ruler" class="w-5 h-5 mr-2"></i>
                        <h3 class="text-lg font-semibold">Valuation Football Field</h3>
                    </div>
                    <div class="w-full overflow-x-auto">
                        <svg width="${{chartWidth}}" height="${{valuationData.length * 40 + 20}}">
                            ${{bars}}
                            <!-- Reference Line for Current Price -->
                            <line x1="${{priceLineX}}" y1="10" x2="${{priceLineX}}" y2="${{valuationData.length * 40 + 10}}" stroke="#f43f5e" stroke-width="2" stroke-dasharray="4 2"></line>
                            <text x="${{priceLineX}}" y="10" fill="#f43f5e" font-size="10" text-anchor="middle" class="font-mono">Live Price: ${{currentPrice.toFixed(2)}}</text>
                        </svg>
                    </div>
                `;
            }}

            // --- 4. Risk Matrix ---
            function renderRiskMatrix() {{
                const container = document.getElementById('risk-matrix');
                const risks = data.key_risks || [];

                const colorScale = [
                    ['#059669', '#10b981', '#34d399', '#6ee7b7', '#a7f3d0'], // Emerald
                    ['#f59e0b', '#facc15', '#fde047', '#fef08a', '#fef9c3'], // Amber
                    ['#dc2626', '#ef4444', '#f87171', '#fca5a5', '#fecaca']  // Rose
                ];

                let gridCells = '';
                for (let i = 5; i >= 1; i--) {{ // Likelihood (rows)
                    for (let j = 1; j <= 5; j++) {{ // Impact (cols)
                        const severity = Math.ceil((i + j) / 2);
                        let color = '#334155'; // Default slate
                        if (severity <= 2) color = colorScale[0][2]; // Low
                        else if (severity <= 4) color = colorScale[1][1]; // Medium
                        else color = colorScale[2][1]; // High

                        const riskAtCell = risks.find(r => r.impact === j && r.likelihood === i);
                        const dot = riskAtCell ? `<div class="risk-dot" title="${{riskAtCell.text}}"></div>` : '';

                        gridCells += `<div class="relative w-full h-full flex items-center justify-center" style="background-color: ${{color}}; opacity: 0.6;">${{dot}}</div>`;
                    }}
                }}

                container.innerHTML = `
                    <div class="flex items-center text-slate-400 mb-4 self-start">
                        <i data-lucide="shield-alert" class="w-5 h-5 mr-2"></i>
                        <h3 class="text-lg font-semibold">Risk Matrix</h3>
                    </div>
                    <div class="flex-grow w-full flex flex-col items-center justify-center">
                        <div class="w-full max-w-[250px] aspect-square grid grid-cols-5 grid-rows-5 gap-1 p-1 bg-slate-900 rounded-md">
                            ${{gridCells}}
                        </div>
                        <div class="flex justify-between w-full max-w-[250px] mt-1 px-1">
                            <span class="text-xs text-slate-500">Low Impact</span>
                            <span class="text-xs text-slate-500">High Impact</span>
                        </div>
                    </div>
                     <div class="text-xs text-slate-500 -rotate-90 absolute left-0 top-1/2 -translate-x-1/2">Likelihood</div>
                `;
            }}

            // --- Initial Render ---
            function renderDashboard() {{
                renderAlphaScore();
                renderInvestmentThesis();
                renderValuationField();
                renderRiskMatrix();
                lucide.createIcons(); // Initialize Lucide icons
            }}

            renderDashboard();
        </script>
    </body>
    </html>
    """

    st.components.v1.html(html_template, height=950, scrolling=True)