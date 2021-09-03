from abc import ABCMeta, abstractmethod

class Component(metaclass=ABCMeta):
    """Abstract class for all callables that could be used in Chainer's pipe."""
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def destroy(self):
        attr_list = list(self.__dict__.keys())
        for attr_name in attr_list:
            attr = getattr(self, attr_name)
            if hasattr(attr, 'destroy'):
                attr.destroy()
            delattr(self, attr_name)

    def serialize(self):
        from serializable import Serializable
        if isinstance(self, Serializable):
            print(f'Method for {self.__class__.__name__} serialization is not implemented!'
                        f' Will not be able to load without using load_path')
        return None

    def deserialize(self, data):
        from serializable import Serializable
        if isinstance(self, Serializable):
            print(f'Method for {self.__class__.__name__} deserialization is not implemented!'
                        f' Please, use traditional load_path for this component')
        pass
