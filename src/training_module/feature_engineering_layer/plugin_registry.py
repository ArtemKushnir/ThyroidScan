from typing import Any, Callable, Type

from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


class TransformRegistry:
    _plugins: dict[str, Type[TransformPlugin]] = {}

    @classmethod
    def register_plugin(cls, name: str) -> Callable[[Type[TransformPlugin]], Type[TransformPlugin]]:
        def decorator(plugin_cls: Type[TransformPlugin]) -> Type[TransformPlugin]:
            if name in cls._plugins:
                raise ValueError(f"Plugin '{name}' already registered")
            cls._plugins[name] = plugin_cls
            return plugin_cls

        return decorator

    @classmethod
    def get_plugin(cls, name: str, *args: Any, **kwargs: Any) -> TransformPlugin:
        if name not in cls._plugins:
            raise ValueError(f"Plugin {name} is not registered")
        return cls._plugins[name](*args, **kwargs)
