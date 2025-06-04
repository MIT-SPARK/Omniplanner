from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable, List, Set, TypeVar, Union, overload

from plum import dispatch, parametric


class FunctorTrait:
    pass


Functor = Union[FunctorTrait, Iterable]


# This function will do type inferences properly for dataclass in plum,
# but I don't think we actually need that functionality right now?
def generic_inference(K):
    @classmethod
    def inference_function(self, *args):
        type_params = K.__type_params__
        field_types = [v.type for v in K.__dataclass_fields__.values()]

        type_bindings = {}
        for arg, typ in zip(args, field_types):
            if issubclass(type(typ), TypeVar):
                if typ in type_bindings:
                    if issubclass(type(arg), type_bindings[typ]):
                        continue
                    if issubclass(type_bindings[typ], type(arg)):
                        type_bindings[typ] = type(arg)
                    else:
                        raise Exception(
                            f"Incompatible concrete types ({type(arg)} vs {type_bindings[typ]}) for generic type {typ}"
                        )
                else:
                    type_bindings[typ] = type(arg)

        concrete_type = ()
        for t in type_params:
            if t in type_bindings:
                concrete_type += (type_bindings[t],)
            else:
                raise Exception(f"No field supplied concrete type for {t}")

        return concrete_type

    setattr(K, "__infer_type_parameter__", inference_function)
    return K


def dispatchable_parametric(f):
    return parametric(generic_inference(dataclass(f)))


@overload
@dispatch
def fmap(fn: Callable, iterable: Iterable):
    raise Exception(
        f"fmap not implemented for {type(iterable)}, but you can implement it!"
    )


@overload
@dispatch
def fmap(fn: Callable, lst: List):
    return [fn(e) for e in lst]


@dispatch
def fmap(fn: Callable, s: Set):
    return set(fn(e) for e in s)


if __name__ == "__main__":

    @dispatchable_parametric
    class GenericTest1[T]:
        name: str
        value: T

    @overload
    @dispatch
    def test(val: GenericTest1[int]):
        print("Generic Test with concrete int")

    @overload
    @dispatch
    def test(val: GenericTest1[str]):
        print("Generic Test with concrete str")

    @overload
    @dispatch
    def test(val: GenericTest1):
        print("Generic test with fallback type")

    @dispatch
    def test(val):
        print("test with fallback type")
