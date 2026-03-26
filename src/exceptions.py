class VisionError(Exception):
    pass


class CameraError(VisionError):
    pass


class ModelError(VisionError):
    pass


class OutputError(VisionError):
    pass


class ConfigurationError(VisionError):
    pass
