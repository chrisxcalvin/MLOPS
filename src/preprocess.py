"""Create data v2: cleaned dataset (dedupe, IQR outlier clipping, drop redundant features)."""
import pandas as pd, numpy as np, sys
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
n0 = len(df)
df = df.drop_duplicates().reset_index(drop=True)
num = df.drop(columns="Class").columns
q1, q3 = df[num].quantile(.25), df[num].quantile(.75)
iqr = q3 - q1
df[num] = df[num].clip(q1 - 3*iqr, q3 + 3*iqr, axis=1)
# drop features highly collinear with another (|r|>0.98) -> ConvexArea, EquivDiameter, Perimeter etc.
corr = df[num].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
drop = [c for c in upper.columns if (upper[c] > 0.98).any()]
df = df.drop(columns=drop)
df.to_csv(dst, index=False)
print(f"rows {n0}->{len(df)}, dropped cols: {drop}, shape {df.shape}")
