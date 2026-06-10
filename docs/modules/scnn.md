# Streaming CNN layers

`lightstream.core.scnn` is the documented public module for streaming CNN layer
building blocks. Import channel layer normalization from this package instead of
private implementation modules:

```python
from lightstream.core.scnn import ChannelLayerNorm, StreamingChannelLayerNorm
```

::: lightstream.core.scnn
