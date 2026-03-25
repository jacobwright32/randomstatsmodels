"""Quick test of statsforecast API."""
import numpy as np
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, AutoTheta, AutoTBATS

y = np.array([10 + 0.05*t + np.sin(2*np.pi*t/12) for t in range(100)])

for name, Model in [("AutoARIMA", AutoARIMA), ("AutoETS", AutoETS),
                     ("AutoCES", AutoCES), ("AutoTheta", AutoTheta),
                     ("AutoTBATS", AutoTBATS)]:
    try:
        m = Model(season_length=12)
        m.fit(y)
        p = m.predict(h=12)
        print(f"{name}: type={type(p).__name__}, shape={getattr(p, 'shape', 'n/a')}")
        vals = p["mean"] if isinstance(p, dict) else p
        print(f"  first 3: {vals[:3]}")
    except Exception as e:
        print(f"{name}: FAILED - {e}")
