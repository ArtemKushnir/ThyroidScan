import importlib
import inspect
import os
import pkgutil
from typing import Any, Callable, Type

from src.training_module.model_core.base_models import BaseModel


class ModelRegistry:
    _models: dict[str, Type["BaseModel"]] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            if name in cls._models:
                raise ValueError(f"Model '{name}' already registered")
            cls._models[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get_model(cls, name: str, **kwargs: Any) -> BaseModel:
        if name not in cls._models:
            raise ValueError(f"Model {name} is not registered")
        return cls._models[name](**kwargs)

    @classmethod
    def discover_models(cls) -> None:
        """
        Discover and register all models in the models directory.
        """
        if cls._initialized:
            return

        plugin_pkg = "src.training_module.model_core.models"

        try:
            package = importlib.import_module(plugin_pkg)
            package_dir = os.path.dirname(inspect.getfile(package))

            for _, name, is_pkg in pkgutil.iter_modules([package_dir]):
                if name == "__init__" or name == "hybrid_blocks":
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
