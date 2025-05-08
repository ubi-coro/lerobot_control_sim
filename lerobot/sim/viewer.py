from abc import ABC, abstractmethod
import cv2
import mujoco.viewer
import logging
import numpy as np


def create_viewer(key, **kwargs):
    return AbstractViewer.create_viewer(key, **kwargs)

class ViewerRegistry:
    _registry = {}

    @classmethod
    def register_subclass(cls, key):
        def decorator(subclass):
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create_viewer(cls, key, **kwargs):
        if key not in cls._registry:
            raise ValueError(f"Unknown viewer type: {key}")
        return cls._registry[key](**kwargs)


class AbstractViewer(ViewerRegistry, ABC):
    def __enter__(self): return self

    def __exit__(self, exc_type, exc_val, exc_tb): pass

    @abstractmethod
    def is_running(self): pass

    @abstractmethod
    def sync(self, observation): pass


@AbstractViewer.register_subclass("mujoco")
class MujocoViewer(AbstractViewer):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data
        self.viewer = None

        if kwargs:
            logging.debug(f"Unused parameters in MujocoViewer: {kwargs}")


    def __enter__(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
        self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0
        self.viewer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.viewer.__exit__(exc_type, exc_val, exc_tb)

    def is_running(self):
        return self.viewer.is_running()

    def sync(self, observation):
        self.viewer.sync()


@AbstractViewer.register_subclass("camera")
class CVViewer(AbstractViewer):
    def __init__(self, image_keys=None, **kwargs):
        self.image_keys = image_keys
        self.running = True

        if kwargs:
            logging.debug(f"Unused parameters in CVViewer: {kwargs}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()
        self.running = False

    def is_running(self):
        return self.running

    def sync(self, observation):
        if not self.image_keys:
            self.image_keys = list(observation['pixels'].keys())
        for key in self.image_keys:
            image = observation['pixels'][key]
            # Shape: (1, 480, 640, 3) -> (3, 480, 640)
            image = np.transpose(image.squeeze(0), (0, 1, 2))
            cv2.imshow(key, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # close with Esc
        if cv2.waitKey(1) == 27:
            self.running = False

@AbstractViewer.register_subclass("headless")
class HeadlessViewer(AbstractViewer):
    def __init__(self, **kwargs):
        self.running = True
        if kwargs:
            logging.debug(f"Unused parameters in HeadlessViewer: {kwargs}")


    def is_running(self):
        return self.running

    def sync(self, observation):
        pass