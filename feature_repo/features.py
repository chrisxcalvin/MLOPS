from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64

bean = Entity(name="bean", join_keys=["bean_id"])
src = FileSource(path="data/beans.parquet", timestamp_field="event_timestamp")
bean_features = FeatureView(
    name="bean_features", entities=[bean], ttl=timedelta(days=365), source=src,
    schema=[Field(name="Area", dtype=Float64), Field(name="AspectRation", dtype=Float64), Field(name="ShapeFactor1", dtype=Float64)],
)
