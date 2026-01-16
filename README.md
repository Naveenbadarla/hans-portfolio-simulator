# Hans Portfolio Walk-Forward Simulator (Streamlit)

A user-friendly simulator for a German (15-min) residential portfolio (default 1000 customers) with EV/PV/Battery.
Workflow: Meter history → profiling → forecasting → forwards hedge → DA schedule → intraday loop → settlement.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
