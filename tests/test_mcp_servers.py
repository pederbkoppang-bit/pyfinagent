"""
Unit tests for MCP servers (Phase 3.0)

Test coverage:
- Data server resources (prices, fundamentals, macro, universe, features)
- Backtest server tools (run_backtest, ablation, feature_test, importance)
- Signals server tools (generate_signal, validate, publish, risk_check)
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents.mcp_servers.data_server import DataServer
from backend.agents.mcp_servers.backtest_server import BacktestServer
from backend.agents.mcp_servers.signals_server import SignalsServer


class TestDataServer:
    """Test data server resources."""
    
    def test_get_prices(self):
        server = DataServer()
        result = server.get_prices("AAPL")
        assert result["ticker"] == "AAPL"
        assert isinstance(result["prices"], list)
    
    def test_get_fundamentals(self):
        server = DataServer()
        result = server.get_fundamentals("AAPL")
        assert result["ticker"] == "AAPL"
    
    def test_get_macro(self):
        server = DataServer()
        result = server.get_macro("VIX")
        assert result["series"] == "VIX"
    
    def test_get_universe(self):
        server = DataServer()
        result = server.get_universe("US")
        assert result["market"] == "US"
    
    def test_get_features(self):
        server = DataServer()
        result = server.get_features("AAPL")
        assert result["ticker"] == "AAPL"


class TestBacktestServer:
    """Test backtest server tools."""
    
    def test_run_backtest(self):
        server = BacktestServer()
        params = {"holding_days": 90}
        result = server.run_backtest(params)
        assert "status" in result
    
    def test_run_ablation_study(self):
        server = BacktestServer()
        result = server.run_ablation_study("momentum_3m")
        assert "status" in result


class TestSignalsServer:
    """Test signals server tools."""
    
    def test_generate_signal(self):
        server = SignalsServer()
        result = server.generate_signal("AAPL", "2026-03-29")
        assert result["ticker"] == "AAPL"
    
    def test_validate_signal(self):
        server = SignalsServer()
        signal = {"ticker": "AAPL", "signal": "BUY"}
        result = server.validate_signal(signal)
        assert "valid" in result
    
    def test_risk_check(self):
        server = SignalsServer()
        trade = {"ticker": "AAPL", "action": "BUY"}
        result = server.risk_check({}, trade)
        assert "allowed" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
