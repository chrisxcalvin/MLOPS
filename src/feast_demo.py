import pandas as pd
from feast import FeatureStore
fs = FeatureStore("feature_repo")
feats = ["bean_features:Area", "bean_features:AspectRation", "bean_features:ShapeFactor1"]
# TRAINING: point-in-time correct historical features
ent = pd.DataFrame({"bean_id": [0, 1, 2], "event_timestamp": pd.to_datetime(["2026-10-02"] * 3, utc=True)})
print("--- offline (training) ---"); print(fs.get_historical_features(ent, feats).to_df())
# INFERENCE: low-latency online lookup of the SAME definitions
print("--- online (inference) ---"); print(pd.DataFrame(fs.get_online_features(features=feats, entity_rows=[{"bean_id": 0}, {"bean_id": 1}, {"bean_id": 2}]).to_dict()))
