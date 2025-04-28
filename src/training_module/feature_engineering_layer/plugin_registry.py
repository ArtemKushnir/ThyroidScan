import abc
from typing import Callable, Type

from src.training_module.feature_engineering_layer.plugins import Plugin


class PluginRegistry:
    """Abstract Class for registering, storing, and receiving Plugins"""

    _plugins: dict[str, Type[Plugin]]

    @classmethod
    def register_plugin(cls, name: str) -> Callable[[Type[Plugin]], Type[Plugin]]:
        def decorator(plugin_cls: Type[Plugin]) -> Type[Plugin]:
            if name in cls._plugins:
                raise ValueError(f"Plugin '{name}' already registered")
            cls._plugins[name] = plugin_cls
            return plugin_cls

        return decorator

    @classmethod
    def get_plugin(cls, name: str) -> Plugin:
        if name not in cls._plugins:
            raise ValueError(f"Plugin {name} is not registered")
        return cls._plugins[name]()


class FeatureRegistry(PluginRegistry):
    _plugins = {}


class TransformRegistry(PluginRegistry):
    _plugins = {}
