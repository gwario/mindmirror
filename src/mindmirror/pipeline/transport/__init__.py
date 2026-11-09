"""Transport connectors and MessageBus implementations."""

import os
from typing import Any

from mindmirror.pipeline.transport.interface import MessageBus
from mindmirror.pipeline.transport.local import LocalBus
from mindmirror.pipeline.transport.zmq_bus import ZmqBus

__all__ = ["MessageBus", "LocalBus", "ZmqBus", "create_message_bus"]

# Optional connectors
try:
    from mindmirror.pipeline.transport.nats_bus import NatsBus
    __all__.append("NatsBus")
except ImportError:
    pass

try:
    from mindmirror.pipeline.transport.valkey_bus import ValkeyBus
    __all__.append("ValkeyBus")
except ImportError:
    pass


def create_message_bus(bus_type: str | None = None, **kwargs: Any) -> MessageBus:
    """Factory creating configured MessageBus instance based on bus_type or BUS_TYPE environment variable."""
    if bus_type is None:
        bus_type = os.getenv("BUS_TYPE", "local").lower()

    if bus_type == "nats":
        from mindmirror.pipeline.transport.nats_bus import NatsBus
        nats_url = kwargs.get("nats_url") or os.getenv("NATS_URL", "nats://127.0.0.1:4222")
        return NatsBus(servers=nats_url)
    elif bus_type in ("valkey", "redis"):
        from mindmirror.pipeline.transport.valkey_bus import ValkeyBus
        valkey_url = kwargs.get("url") or os.getenv("VALKEY_URL", "redis://127.0.0.1:6379/0")
        return ValkeyBus(url=valkey_url)
    elif bus_type == "zmq":
        from mindmirror.pipeline.transport.zmq_bus import ZmqBus
        return ZmqBus(**kwargs)
    else:
        return LocalBus()
