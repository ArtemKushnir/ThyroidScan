from typing import Callable, Type

from src.training_module.feature_engineering_layer.feature_plugin import FeaturePlugin


class FeatureRegistry:
    """Class for registering, storing, and receiving FeaturePlugin"""

    _plugins: dict[str, Type[FeaturePlugin]] = {}

    @classmethod
    def register_plugin(cls, name: str) -> Callable[[Type[FeaturePlugin]], Type[FeaturePlugin]]:
        def decorator(plugin_cls: Type[FeaturePlugin]) -> Type[FeaturePlugin]:
            if name in cls._plugins:
                raise ValueError(f"Plugin '{name}' already registered")
            cls._plugins[name] = plugin_cls
            return plugin_cls

        return decorator

    @classmethod
    def get_plugin(cls, name: str) -> FeaturePlugin:
        if name not in cls._plugins:
            raise ValueError(f"Plugin {name} is not registered")
        return cls._plugins[name]()
