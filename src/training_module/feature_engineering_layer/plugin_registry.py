from typing import Any, Callable, Type, Union

from src.training_module.feature_engineering_layer.plugins import FeaturePlugin, TransformPlugin


class PluginRegistryMixin:
    """Abstract Class for registering, storing, and receiving Plugins"""

    _plugins: dict[str, Any] = {}

    @classmethod
    def _register_plugin(cls, name: str) -> Callable[[Type], Type]:
        def decorator(plugin_cls: Type) -> Type:
            if name in cls._plugins:
                raise ValueError(f"Plugin '{name}' already registered")
            cls._plugins[name] = plugin_cls
            return plugin_cls

        return decorator

    @classmethod
    def _get_plugin(cls, name: str, **kwargs: Any) -> Any:
        if name not in cls._plugins:
            raise ValueError(f"Plugin {name} is not registered")
        return cls._plugins[name](**kwargs)


class FeatureRegistry(PluginRegistryMixin):
    _plugins: dict[str, FeaturePlugin] = {}

    @classmethod
    def register_plugin(cls, name: str) -> Callable[[Type[FeaturePlugin]], Type[FeaturePlugin]]:
        return cls.register_plugin(name)

    @classmethod
    def get_plugin(cls, name: str) -> FeaturePlugin:
        return cls._get_plugin(name)


class TransformRegistry(PluginRegistryMixin):
    _plugins: dict[str, TransformPlugin] = {}

    @classmethod
    def register_plugin(cls, name: str) -> Callable[[Type[TransformPlugin]], Type[TransformPlugin]]:
        return cls.register_plugin(name)

    @classmethod
    def get_plugin(cls, name: str, **kwargs: Any) -> TransformPlugin:
        return cls.get_plugin(name, **kwargs)
