from tensorboardX import SummaryWriter


class NullLogger:

    def __init__(self):
        pass

    @property
    def run_name(self):
        return "Null"

    def meta(self, *args, **kwargs):
        pass

    def log(self, name, loss, step):
        pass


class TensorboardX:

    def __init__(self, logdir, run_name):
        self._logdir = logdir
        self._writer = SummaryWriter(logdir=logdir)
        self.run_name = run_name

    def args(self, arg_text):
        self._writer.add_text("args", arg_text)

    def meta(self, params):
        self._writer.add_hparams(hparam_dict=params, metric_dict={})

    def log(self, name, loss, step):
        self._writer.add_scalar(name, loss, step)


class CometML:

    def __init__(self):
        pass

    @property
    def run_name(self):
        return "Null"

    def meta(self, *args, **kwargs):
        pass

    def log(self, name, loss, step):
        pass


class WanDB:

    def __init__(self):
        pass

    @property
    def run_name(self):
        return "Null"

    def meta(self, *args, **kwargs):
        pass

    def log(self, name, loss, step):
        pass
