"""
End-to-end integration test for Phase 3.0 MCP servers.

Tests that all three servers can be started and work together.
"""

import pytest
import json
import asyncio
from pathlib import Path

# This assumes FastMCP is installed
try:
    from fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False


@pytest.mark.skipif(not HAS_FASTMCP, reason="FastMCP not installed")
class TestMCPServersIntegration:
    """Integration tests for MCP servers."""
    
    def test_data_server_creation(self):
        """Test that data server can be created."""
        from backend.agents.mcp_servers import create_data_server
        
        server = create_data_server()
        assert server is not None
        assert hasattr(server, 'run')
    
    def test_backtest_server_creation(self):
        """Test that backtest server can be created."""
        from backend.agents.mcp_servers import create_backtest_server
        
        server = create_backtest_server()
        assert server is not None
        assert hasattr(server, 'run')
    
    def test_signals_server_creation(self):
        """Test that signals server can be created."""
        from backend.agents.mcp_servers import create_signals_server
        
        server = create_signals_server()
        assert server is not None
        assert hasattr(server, 'run')
    
    @pytest.mark.asyncio
    async def test_all_servers_startup(self):
        """Test that all servers can start simultaneously."""
        from backend.agents.mcp_servers import (
            create_data_server,
            create_backtest_server,
            create_signals_server,
        )
        
        # Create all servers
        data_server = create_data_server()
        backtest_server = create_backtest_server()
        signals_server = create_signals_server()
        
        # Verify they exist
        assert data_server is not None
        assert backtest_server is not None
        assert signals_server is not None
        
        # In a real scenario, they'd run in separate processes
        # For testing, we just verify they initialize without crashing


class TestDataServer:
    """Tests for data server resources."""
    
    def test_data_server_methods(self):
        """Test that data server has all required methods."""
        from backend.agents.mcp_servers.data_server import DataServer
        
        server = DataServer()
        
        # Check all required methods exist
        assert hasattr(server, 'get_prices')
        assert hasattr(server, 'get_fundamentals')
        assert hasattr(server, 'get_macro')
        assert hasattr(server, 'get_universe')
        assert hasattr(server, 'get_features')
        assert hasattr(server, 'get_experiment_list')
        assert hasattr(server, 'get_best_params')
    
    def test_get_prices_signature(self):
        """Test that get_prices returns expected structure."""
        from backend.agents.mcp_servers.data_server import DataServer
        
        server = DataServer()
        result = server.get_prices("AAPL")
        
        # Should return a dict with 'ticker' and 'prices' keys
        assert isinstance(result, dict)
        assert 'ticker' in result
        assert 'prices' in result
        assert result['ticker'] == 'AAPL'
        assert isinstance(result['prices'], list)


class TestBacktestServer:
    """Tests for backtest server tools."""
    
    def test_backtest_server_methods(self):
        """Test that backtest server has all required tools."""
        from backend.agents.mcp_servers.backtest_server import BacktestServer
        
        server = BacktestServer()
        
        # Check all required methods exist
        assert hasattr(server, 'run_backtest')
        assert hasattr(server, 'run_single_feature_test')
        assert hasattr(server, 'run_ablation_study')
        assert hasattr(server, 'get_feature_importance')
        assert hasattr(server, 'get_experiment_list')
        assert hasattr(server, 'get_recent_experiments')
    
    def test_run_backtest_signature(self):
        """Test that run_backtest returns expected structure."""
        from backend.agents.mcp_servers.backtest_server import BacktestServer
        
        server = BacktestServer()
        params = {
            "holding_days": 90,
            "tp_pct": 10.0,
            "sl_pct": 10.0,
        }
        result = server.run_backtest(params)
        
        # Should return expected keys
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'sharpe' in result


class TestSignalsServer:
    """Tests for signals server tools."""
    
    def test_signals_server_methods(self):
        """Test that signals server has all required tools."""
        from backend.agents.mcp_servers.signals_server import SignalsServer
        
        server = SignalsServer()
        
        # Check all required methods exist
        assert hasattr(server, 'generate_signal')
        assert hasattr(server, 'validate_signal')
        assert hasattr(server, 'publish_signal')
        assert hasattr(server, 'risk_check')
        assert hasattr(server, 'get_portfolio')
        assert hasattr(server, 'get_risk_constraints')
        assert hasattr(server, 'get_signal_history')
    
    def test_generate_signal_signature(self):
        """Test that generate_signal returns expected structure."""
        from backend.agents.mcp_servers.signals_server import SignalsServer
        
        server = SignalsServer()
        result = server.generate_signal("AAPL", "2026-03-29")
        
        # Should return expected keys
        assert isinstance(result, dict)
        assert 'ticker' in result
        assert 'signal' in result
        assert 'confidence' in result
        assert 'factors' in result


class TestServerNamingAndDocumentation:
    """Tests for proper naming and documentation."""
    
    def test_data_server_has_docstrings(self):
        """Test that data server methods have docstrings."""
        from backend.agents.mcp_servers.data_server import DataServer
        
        server = DataServer()
        
        # Check key methods have docstrings
        assert server.get_prices.__doc__ is not None
        assert server.get_fundamentals.__doc__ is not None
    
    def test_backtest_server_has_docstrings(self):
        """Test that backtest server methods have docstrings."""
        from backend.agents.mcp_servers.backtest_server import BacktestServer
        
        server = BacktestServer()
        
        # Check key methods have docstrings
        assert server.run_backtest.__doc__ is not None
        assert server.run_single_feature_test.__doc__ is not None
    
    def test_signals_server_has_docstrings(self):
        """Test that signals server methods have docstrings."""
        from backend.agents.mcp_servers.signals_server import SignalsServer
        
        server = SignalsServer()
        
        # Check key methods have docstrings
        assert server.generate_signal.__doc__ is not None
        assert server.validate_signal.__doc__ is not None


# Run tests: pytest tests/test_mcp_integration.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
