from tensorboardX import SummaryWriter


class NullLogger:

    def __init__(self):
        pass

    def meta(self, *args, **kwargs):
        pass

    def log(self, name, loss, step):
        pass


class TensorboardX:

    def __init__(self, logdir):
        self._logdir = logdir
        self._writer = SummaryWriter(logdir=logdir)

    def meta(self, *args, **kwargs):
        pass

    def log(self, name, loss, step):
        self._writer.add_scalar(name, loss, step)


class CometML:

    def __init__(self):
        pass

    def meta(self, *args, **kwargs):
        pass

    def log(self, name, loss, step):
        pass


class WanDB:

    def __init__(self):
        pass

    def meta(self, *args, **kwargs):
        pass

    def log(self, name, loss, step):
        pass
