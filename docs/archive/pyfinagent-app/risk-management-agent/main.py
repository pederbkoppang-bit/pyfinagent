import os
import logging
import numpy as np
import functions_framework
from pydantic import BaseModel, ValidationError, Field
import redis
from typing import Dict, Any, Optional, List
from google.cloud import bigquery
from datetime import datetime

# Configuration
logging.basicConfig(level=logging.INFO)
# Environment variables must be set in the GCP Cloud Function configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = "pyfinagent_data"
TABLE_ID = "risk_intervention_log"
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
STATE_KEY_PREFIX = "agent_state" # e.g., agent_state:SPY

# Initialize BigQuery Client for Logging (outside handler for connection reuse)
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    # Define table reference string for logging
    table_ref_str = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
except Exception as e:
    logging.error(f"Failed to initialize BigQuery client: {e}")
    bq_client = None
    table_ref_str = None

# Initialize Redis Client
try:
    if not REDIS_HOST:
        raise ValueError("REDIS_HOST environment variable not set.")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping() # Check connection
except Exception as e:
    logging.error(f"Failed to initialize Redis client: {e}")
    redis_client = None

# --- Pydantic Models for Input Validation ---
# Ensures the incoming State data structure is valid before risk evaluation.

class PortfolioState(BaseModel):
    equity: float = Field(..., gt=0)
    current_drawdown: float = Field(..., ge=0)
    gross_exposure: float = Field(..., ge=0)

class MicrostructureState(BaseModel):
    # Normalized OFI (Ref 2.1)
    OFI_1m_norm: Optional[float] = None 
    # Raw L1 depth for liquidity checks
    L1_depth_bid: Optional[int] = Field(None, ge=0)
    L1_depth_ask: Optional[int] = Field(None, ge=0)
    # VEX: Measures delta change per 1% change in IV.
    vanna_exposure_vex: Optional[float] = None
    # CEX: Measures daily delta decay.
    charm_exposure_cex: Optional[float] = None
    # Speed: Measures the rate of change of Gamma.
    speed: Optional[float] = None

class InnovationState(BaseModel):
    patent_velocity_pct: Optional[float] = None
    citation_lag_days: Optional[int] = None
    new_filings_count: Optional[int] = None

class LaborMomentumState(BaseModel):
    rd_job_growth_pct: Optional[float] = None
    specialized_role_count: Optional[int] = None

class MacroState(BaseModel):
    VIX: Optional[float] = Field(None, gt=0)

class BehavioralState(BaseModel):
    # Options Skew (Ref 4.3)
    IV_skew_25delta: Optional[float] = None
    # Historical baseline for skew
    IV_skew_14d_avg: Optional[float] = None

class MemoryState(BaseModel):
    # Must be Annualized Volatility Forecast (Ref 5.3)
    sigma_GARCH_forecast: Optional[float] = Field(None, gt=0)

# Simplified State Input for the Risk Manager (Aligns with BQ Schema structure)
class AgentState(BaseModel):
    portfolio: PortfolioState
    microstructure: MicrostructureState
    macro: MacroState
    behavioral: BehavioralState
    memory: MemoryState
    innovation: Optional[InnovationState] = None
    labor: Optional[LaborMomentumState] = None

class ProposedAction(BaseModel):
    event_id: str
    ticker: str
    agent_id: str
    action_type: str = Field(..., pattern="^(BUY|SELL)$")
    proposed_size: int = Field(..., gt=0)
    price: float = Field(..., gt=0) # Arrival Price

class PredictiveSignals(BaseModel):
    """Holds signals derived from market microstructure and options data."""
    accumulation_alert: Optional[str] = None
    bid_ask_depth_ratio: Optional[float] = None
    skew_flattening_pct: Optional[float] = None

class ApprovedAction(BaseModel):
    status: str # APPROVED, ADJUSTED, VETO
    approved_size: int
    risk_intervention: bool
    reasons: List[str]
    predictive_signals: PredictiveSignals
    event_id: str

# --- The Risk Gatekeeper ---

class RiskGatekeeper:
    def __init__(self):
        # Load thresholds (Ideally from environment variables)
        self.RISK_TARGET_FRACTION = float(os.getenv("RMA_RISK_TARGET_FRACTION", 0.01))
        self.MAX_DRAWDOWN = float(os.getenv("RMA_MAX_DRAWDOWN", 0.15))
        
        # Macro/Tail Risk
        self.VIX_HALT = float(os.getenv("RMA_VIX_HALT", 35.0))
        self.VIX_WARNING = float(os.getenv("RMA_VIX_WARNING", 25.0))
        self.SKEW_FEAR_THRESHOLD = float(os.getenv("RMA_SKEW_FEAR_THRESHOLD", 1.5))
        
        # Microstructure (Ref 2.1)
        self.OFI_TOXICITY_THRESHOLD = float(os.getenv("RMA_OFI_TOXICITY_THRESHOLD", 5.0))
        self.MAX_L1_PARTICIPATION = float(os.getenv("RMA_MAX_L1_PARTICIPATION", 0.10))
        
        # Hard Constraints
        self.MAX_CONCENTRATION = float(os.getenv("RMA_MAX_CONCENTRATION", 0.20))
        self.MAX_GROSS_LEVERAGE = float(os.getenv("RMA_MAX_GROSS_LEVERAGE", 1.5))

        # Predictive Accumulation Thresholds
        self.SKEW_FLATTENING_THRESHOLD = float(os.getenv("RMA_SKEW_FLATTENING_THRESHOLD", 0.12)) # 12% drop
        self.BID_ASK_RATIO_THRESHOLD = float(os.getenv("RMA_BID_ASK_RATIO_THRESHOLD", 2.5))
        self.RELATIVE_VOLUME_THRESHOLD = float(os.getenv("RMA_RELATIVE_VOLUME_THRESHOLD", 1.5)) # 50% spike

    def evaluate(self, state: AgentState, action: ProposedAction) -> ApprovedAction:
        reasons = []
        
        # 1. Hard Circuit Breakers (Drawdown)
        if state.portfolio.current_drawdown > self.MAX_DRAWDOWN:
            return self._finalize_decision(action, 0, [f"VETO: Max Drawdown Exceeded ({state.portfolio.current_drawdown*100:.2f}%)."])

        # 2. Volatility-Targeted Sizing (Ref: Research 5.3)
        # Calculate the maximum allowable size based on risk budget and volatility.
        max_allowable_size, sizing_reasons = self._calculate_vol_adjusted_size(state, action.price)
        reasons.extend(sizing_reasons)

        if max_allowable_size == 0:
            return self._finalize_decision(action, 0, reasons)

        # Determine starting size: minimum of proposed and allowable
        current_size = min(action.proposed_size, max_allowable_size)
        if current_size < action.proposed_size:
            reasons.append(f"ADJUSTED: Volatility Target Cap. Reduced size to {current_size}.")

        # 3. Macro/Tail Risk Adjustment (Ref: Research 4.3)
        macro_multiplier, macro_reasons = self._evaluate_macro_risk(state)
        current_size = int(current_size * macro_multiplier)
        reasons.extend(macro_reasons)

        if macro_multiplier == 0:
            return self._finalize_decision(action, 0, reasons)

        # 4. Microstructure Checks (Toxicity and Liquidity) (Ref: Research 2.1)
        current_size, micro_reasons, toxic_veto = self._evaluate_microstructure(state.microstructure, action, current_size)
        reasons.extend(micro_reasons)

        if toxic_veto:
            return self._finalize_decision(action, 0, reasons)

        # 5. Hard Constraints (Concentration and Leverage)
        current_size, constraint_reasons = self._check_hard_constraints(current_size, state.portfolio, action.price)
        reasons.extend(constraint_reasons)

        # 6. NEW: Detect Institutional Accumulation
        predictive_signals = self._detect_institutional_accumulation(state)

        return self._finalize_decision(action, current_size, reasons, predictive_signals)

    def _calculate_vol_adjusted_size(self, state: AgentState, price: float):
        # Uses GARCH forecast (Annualized) for volatility targeting
        annualized_vol = state.memory.sigma_GARCH_forecast
        if not annualized_vol or annualized_vol <= 0:
            return 0, ["VETO: GARCH forecast unavailable or invalid."]
        
        # Convert annualized volatility to daily volatility (sqrt(252))
        daily_vol_pct = annualized_vol / np.sqrt(252)
        # Define risk per share (e.g., stop loss at a 2-sigma daily move)
        risk_per_share = price * daily_vol_pct * 2
        
        if risk_per_share <= 0:
             return 0, ["VETO: Invalid risk per share calculation."]

        risk_budget = state.portfolio.equity * self.RISK_TARGET_FRACTION
        size = int(risk_budget / risk_per_share)
        return size, [f"Vol Target Check: Max Allowable Size {size}."]

    def _evaluate_macro_risk(self, state: AgentState):
        # Checks VIX and Skew
        multiplier = 1.0
        reasons = []
        vix = state.macro.VIX
        skew = state.behavioral.IV_skew_25delta

        if vix:
            if vix > self.VIX_HALT:
                return 0.0, [f"VETO: VIX Halt ({vix:.2f})."]
            elif vix > self.VIX_WARNING:
                multiplier *= 0.75
                reasons.append(f"ADJUST: VIX Warning ({vix:.2f}). Size reduced 25%.")

        # Reduce exposure if Skew indicates high fear (Ref 4.3)
        if skew and skew > self.SKEW_FEAR_THRESHOLD:
            multiplier *= 0.5
            reasons.append(f"ADJUST: Tail Risk (Skew={skew:.2f}). Size reduced 50%.")
        return multiplier, reasons

    def _evaluate_microstructure(self, micro: MicrostructureState, action: ProposedAction, size: int):
        # Checks OFI Toxicity and L1 Depth Liquidity (Ref 2.1)
        ofi = micro.OFI_1m_norm
        reasons = []

        # Toxicity Check (Adverse Selection)
        if ofi is not None:
            if (action.action_type == 'BUY' and ofi < -self.OFI_TOXICITY_THRESHOLD) or \
               (action.action_type == 'SELL' and ofi > self.OFI_TOXICITY_THRESHOLD):
                return 0, [f"VETO: Toxic OFI ({ofi:.2f}). Adverse selection risk."], True

        # Liquidity Check (Market Impact)
        # Check depth on the opposite side of the trade
        l1_depth = micro.L1_depth_ask if action.action_type == 'BUY' else micro.L1_depth_bid
        
        if l1_depth is not None and l1_depth > 0:
            liquidity_cap = int(l1_depth * self.MAX_L1_PARTICIPATION)
            if size > liquidity_cap:
                reasons.append(f"ADJUST: Liquidity Cap (Max {self.MAX_L1_PARTICIPATION*100:.0f}% L1). Size reduced to {liquidity_cap}.")
                size = liquidity_cap
        elif l1_depth == 0:
             return 0, [f"VETO: No liquidity available (L1 Depth=0)."], True
        
        return size, reasons, False

    def _check_hard_constraints(self, size, portfolio: PortfolioState, price):
        # Checks Concentration and Leverage limits
        if price <= 0: return 0, ["VETO: Invalid Price."]
        proposed_value = size * price
        reasons = []

        # Concentration
        cap_value = portfolio.equity * self.MAX_CONCENTRATION
        if proposed_value > cap_value:
            cap_size = int(cap_value / price)
            reasons.append(f"ADJUST: Concentration Limit. Capped to {cap_size}.")
            size = cap_size
            proposed_value = size * price

        # Leverage
        new_exposure = portfolio.gross_exposure + proposed_value
        max_exposure = portfolio.equity * self.MAX_GROSS_LEVERAGE
        if new_exposure > max_exposure:
            allowable_value = max_exposure - portfolio.gross_exposure
            if allowable_value > 0:
                cap_size = int(allowable_value / price)
                reasons.append(f"ADJUST: Leverage Limit. Capped to {cap_size}.")
                size = cap_size
            else:
                size = 0
                reasons.append("VETO: Leverage Limit Breached. Cannot increase exposure.")

        return max(0, size), reasons

    def _detect_institutional_accumulation(self, state: AgentState) -> PredictiveSignals:
        """
        Analyzes options skew and L1 depth to detect 'smart money' accumulation.
        Based on research by Xing, Zhang, & Zhao (2010) and Easley et al.
        """
        signals = PredictiveSignals()
        skew_flattened = False
        bid_depth_strong = False
        positive_vanna = False

        # 1. Options Skew Flattening Check
        current_skew = state.behavioral.IV_skew_25delta
        historical_skew = state.behavioral.IV_skew_14d_avg

        if current_skew is not None and historical_skew is not None and historical_skew > 0:
            flattening_pct = (historical_skew - current_skew) / historical_skew
            signals.skew_flattening_pct = round(flattening_pct, 4)
            if flattening_pct >= self.SKEW_FLATTENING_THRESHOLD:
                skew_flattened = True

        # 2. Bid/Ask Depth Ratio Check
        bid_depth = state.microstructure.L1_depth_bid
        ask_depth = state.microstructure.L1_depth_ask

        if bid_depth is not None and ask_depth is not None and ask_depth > 0:
            ratio = bid_depth / ask_depth
            signals.bid_ask_depth_ratio = round(ratio, 2)
            if ratio > self.BID_ASK_RATIO_THRESHOLD:
                bid_depth_strong = True

        # 3. Vanna Exposure Check (Recursive Buying Loop Signal)
        vanna = state.microstructure.vanna_exposure_vex
        if vanna is not None and vanna > 0:
            positive_vanna = True

        # 4. Confluence Signal: Trigger alert if all conditions are met
        if skew_flattened and bid_depth_strong and positive_vanna: # type: ignore
            signals.accumulation_alert = "CRITICAL - Hyper-growth structural setup detected: Skew flattening < 0.12, Bid-Ask > 2.5, and Positive Vanna/Charm exposure."

        # 5. Structural Decoupling Check (Escalation)
        innovation = state.innovation
        labor = state.labor
        if innovation and labor and innovation.patent_velocity_pct is not None and labor.rd_job_growth_pct is not None and innovation.patent_velocity_pct >= 0.20 and labor.rd_job_growth_pct >= 0.30:
            signals.accumulation_alert = "STRUCTURAL OVERDRIVE - Innovation and Labor momentum confirm institutional accumulation phase."

        return signals

    def _finalize_decision(self, action: ProposedAction, approved_size: int, reasons: List[str], predictive_signals: PredictiveSignals) -> ApprovedAction:
        risk_intervention = False
        if approved_size <= 0:
            status = "VETO"
            approved_size = 0
            risk_intervention = True
        # Check if any ADJUSTED/Capping reasons exist
        elif any("ADJUSTED:" in r or "Capping" in r or "Size reduced" in r for r in reasons):
            status = "ADJUSTED"
            risk_intervention = True
        else:
            status = "APPROVED"
        
        response = ApprovedAction(
            status=status, 
            approved_size=approved_size, 
            reasons=reasons, 
            event_id=action.event_id,
            risk_intervention=risk_intervention,
            predictive_signals=predictive_signals
        )
        
        # Log the decision (especially interventions) to BigQuery
        self._log_risk_event(response, action)
            
        return response

    def _log_risk_event(self, response: ApprovedAction, action: ProposedAction):
        """Streams the risk decision details to BigQuery."""
        if not bq_client or not table_ref_str:
            logging.warning("BigQuery client not initialized. Skipping logging.")
            return

        row = {
            "log_timestamp": datetime.utcnow().isoformat(),
            "event_id": response.event_id,
            "agent_id": action.agent_id,
            "ticker": action.ticker,
            "status": response.status,
            "approved_size": response.approved_size,
            "proposed_size": action.proposed_size,
            "reasons": "; ".join(response.reasons) # Combined string for the log
        }
        try:
            errors = bq_client.insert_rows_json(table_ref_str, [row])
            if errors:
                logging.error(f"Error streaming risk log to BigQuery: {errors}")
        except Exception as e:
            logging.error(f"Exception during BigQuery insert: {e}")

# --- Cloud Function Entry Point ---

@functions_framework.http
def risk_gatekeeper_http(request):
    """
    GCP Cloud Function entry point for the Risk Management Gatekeeper.
    """
    request_json = request.get_json(silent=True)

    if not request_json or 'action' not in request_json:
        return {"error": "Invalid input. Must include 'action'."}, 400

    # Validate Inputs using Pydantic V2
    try:
        action_input = ProposedAction.model_validate(request_json['action'])
    except ValidationError as e:
        logging.error(f"Input validation failed: {e.json()}")
        # Fail-Safe: Veto if input data is corrupted or missing critical fields
        event_id = request_json.get('action', {}).get('event_id', 'validation_error')
        return ApprovedAction(status="VETO", approved_size=0, reasons=["VETO: Input Validation Error."], event_id=event_id, risk_intervention=True, predictive_signals=PredictiveSignals()).model_dump(), 400

    # --- NEW: Fetch state independently from Redis ---
    try:
        if not redis_client:
            raise ConnectionError("Redis client not available.")
        state_key = f"{STATE_KEY_PREFIX}:{action_input.ticker}"
        state_json = redis_client.get(state_key)
        if not state_json:
            raise ValueError(f"State not found in Redis for key: {state_key}")
        
        # Validate the state fetched from Redis
        state_input = AgentState.model_validate_json(state_json)

    except (ConnectionError, ValueError, ValidationError) as e:
        logging.error(f"Failed to fetch or validate state from Redis: {e}")
        return ApprovedAction(status="VETO", approved_size=0, reasons=["VETO: Failed to retrieve valid state."], event_id=action_input.event_id, risk_intervention=True, predictive_signals=PredictiveSignals()).model_dump(), 500

    gatekeeper = RiskGatekeeper()

    try:
        approved_action = gatekeeper.evaluate(state_input, action_input)
        return approved_action.model_dump(), 200
    except Exception as e:
        logging.error(f"An unexpected error occurred during risk evaluation: {e}", exc_info=True)
        # Fail-Safe: If the risk engine fails, halt trading.
        return ApprovedAction(status="VETO", approved_size=0, reasons=["VETO: Internal Risk Engine Error."], event_id=action_input.event_id, risk_intervention=True, predictive_signals=PredictiveSignals()).model_dump(), 500