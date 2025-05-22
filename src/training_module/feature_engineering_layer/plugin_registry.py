import importlib
import inspect
import os
import pkgutil
from typing import Any, Callable, Type

from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


class TransformRegistry:
    _plugins: dict[str, Type[TransformPlugin]] = {}
    _initialized: bool = False

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

    @classmethod
    def discover_plugins(cls) -> None:
        """
        Discover and register all plugins in the plugins directory.
        """
        if cls._initialized:
            return

        plugin_pkg = "src.training_module.feature_engineering_layer.plugins"

        try:
            package = importlib.import_module(plugin_pkg)
            package_dir = os.path.dirname(inspect.getfile(package))

            for _, name, is_pkg in pkgutil.iter_modules([package_dir]):
                if name == "__init__" or name == "transform_plugin":
                    continue

                importlib.import_module(f"{plugin_pkg}.{name}")

                if is_pkg:
                    cls._import_submodules(f"{plugin_pkg}.{name}")

            cls._initialized = True
        except ImportError as e:
            print(f"Error discovering plugins: {e}")

    @classmethod
    def _import_submodules(cls, package_name: str) -> None:
        """
        Recursively import all submodules of a package.
        """
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return

        package_dir = os.path.dirname(inspect.getfile(package))
        for _, name, is_pkg in pkgutil.iter_modules([package_dir]):
            if name == "__init__":
                continue

            module_name = f"{package_name}.{name}"
            importlib.import_module(module_name)

            if is_pkg:
                cls._import_submodules(module_name)
