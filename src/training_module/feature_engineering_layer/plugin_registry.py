from typing import Callable, Type, Any

from src.training_module.feature_engineering_layer.transform_plugins import TransformPlugin


class TransformRegistry:
    _plugins: dict[str, TransformPlugin] = {}

    @classmethod
    def register_plugin(cls, name: str) -> Callable[[Type[TransformPlugin]], Type[TransformPlugin]]:
        return cls.register_plugin(name)

    @classmethod
    def get_plugin(cls, name: str, **kwargs: Any) -> TransformPlugin:
        return cls.get_plugin(name, **kwargs)
