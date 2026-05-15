from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class HebbianCell:
    value: float
    strength: float = 0.0


@dataclass
class HebbianTable:
    eta: float = 0.12
    decay: float = 0.002
    _channels: Dict[str, Dict[Tuple[int, int], HebbianCell]] = field(default_factory=dict)

    def _channel(self, op: str) -> Dict[Tuple[int, int], HebbianCell]:
        if op not in self._channels:
            self._channels[op] = {}
        return self._channels[op]

    def predict(self, op: str, a: int, b: int) -> tuple[float | None, float]:
        cell = self._channel(op).get((a, b))
        if cell is None:
            return None, 0.0
        return cell.value, max(0.0, min(1.0, cell.strength))

    def update(self, op: str, a: int, b: int, target: float, precision_gate: float) -> None:
        channel = self._channel(op)
        key = (a, b)
        cell = channel.get(key)
        lr = self.eta * max(0.0, min(1.0, precision_gate))
        if cell is None:
            channel[key] = HebbianCell(value=target, strength=lr)
            return

        cell.value = (1.0 - lr) * cell.value + lr * target
        cell.strength = min(1.0, cell.strength + lr)

    def apply_decay(self) -> None:
        for channel in self._channels.values():
            for key, cell in list(channel.items()):
                cell.strength = max(0.0, cell.strength - self.decay)
                if cell.strength <= 0.0:
                    del channel[key]

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "eta": self.eta,
            "decay": self.decay,
            "channels": {},
        }
        channels: Dict[str, Dict[str, Dict[str, float]]] = {}
        for op, mapping in self._channels.items():
            op_map: Dict[str, Dict[str, float]] = {}
            for (a, b), cell in mapping.items():
                op_map[f"{a},{b}"] = {"value": cell.value, "strength": cell.strength}
            channels[op] = op_map
        data["channels"] = channels
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "HebbianTable":
        table = cls(
            eta=float(data.get("eta", 0.12)),
            decay=float(data.get("decay", 0.002)),
        )
        raw_channels = data.get("channels", {})
        if isinstance(raw_channels, dict):
            for op, mapping in raw_channels.items():
                if not isinstance(op, str) or not isinstance(mapping, dict):
                    continue
                channel: Dict[Tuple[int, int], HebbianCell] = {}
                for key, raw_cell in mapping.items():
                    if not isinstance(key, str) or not isinstance(raw_cell, dict):
                        continue
                    parts = key.split(",")
                    if len(parts) != 2:
                        continue
                    try:
                        a = int(parts[0])
                        b = int(parts[1])
                        value = float(raw_cell.get("value", 0.0))
                        strength = float(raw_cell.get("strength", 0.0))
                    except (TypeError, ValueError):
                        continue
                    channel[(a, b)] = HebbianCell(value=value, strength=strength)
                table._channels[op] = channel
        return table
