from mllib.runners.base_runners import BaseRunner
from mllib.tasks.base_tasks import MNISTMLP


task = MNISTMLP()
runner = BaseRunner(task)
runner.create_trainer()
runner.train()
runner.test()