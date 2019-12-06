from tensorboardX import SummaryWriter
from comet_ml import Experiment
import wandb


class NullLogger:

    def __init__(self):
        pass

    @property
    def run_name(self):
        return "Null"

    def meta(self, *args, **kwargs):
        print(args)
        print(kwargs)

    def log(self, name, loss, step):
        if step % 100 == 0:
            print(f"Step {step} {name}: {loss}")

    def args(self, arg_text):
        print(arg_text)


class TensorboardX:

    def __init__(self, logdir, run_name):
        self._logdir = logdir
        self._writer = SummaryWriter(logdir=logdir)
        self.run_name = run_name

    def args(self, arg_text):
        self._writer.add_text("args", arg_text)

    def meta(self, params):
        self._writer.add_hparams(hparam_dict=params, metric_dict={})

    def log(self, name, value, step):
        self._writer.add_scalar(name, value, step)


class CometML:

    def __init__(self, api_key, project_name, workspace, debug=True, tags=None):
        self._exp = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            disabled=debug,
        )
        if not (self._exp.alive or debug):
            raise RuntimeError("Cannot connect to Comet ML")
        self._exp.disable_mp()

        if tags is not None:
            self._exp.add_tags(tags)

    @property
    def run_name(self):
        return self._exp.get_key()

    def args(self, arg_text):
        self._exp.log_parameter("cmd args", arg_text)

    def meta(self, params):
        self._exp.log_parameters(params)

    def log(self, name, value, step):
        self._exp.log_metric(
            name=name,
            value=value,
            step=step,
        )


class WandB:

    def __init__(self, tags):
        self._run = wandb.init(tags=tags, job_type="training")

    @property
    def run_name(self):
        return self._run.id

    def args(self, arg_text):
        wandb.config.update({"cmd args": arg_text})

    def meta(self, params):
        wandb.config.update(params)

    def log(self, name, value, step):
        wandb.log(
            {name: value},
            step=step,
        )
