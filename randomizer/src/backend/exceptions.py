"""自定义异常类."""


class RandomizerError(Exception):
    """基础异常类."""
    pass


class ConfigError(RandomizerError):
    """配置错误."""
    pass


class ModelError(RandomizerError):
    """模型错误."""
    pass


class GenerationError(RandomizerError):
    """生成错误."""
    pass 