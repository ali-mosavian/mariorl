import time

from os import PathLike
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from pathlib import Path
from operator import itemgetter as get
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.ion()
plt.style.use("seaborn-v0_8")

AggFunc = Callable[[Union[List, np.array]], float]


class MetricLoggerPD:
    def __init__(self, save_dir: PathLike, agg_func: Dict[str, AggFunc]):
        self.path = Path(save_dir)
        self.episodes = list()
        self.curr_ep = defaultdict(list)
        self.agg_func = agg_func

        log = self.path / "log.csv"
        if log.exists():
            self.episodes = pd.read_csv(log).to_dict("records")

    def log_step(self, **kwargs: Dict[str, float]):
        for k, v in kwargs.items():
            if v is None or v is float("nan"):
                continue
            self.curr_ep[k].append(v)

    def total_steps(self) -> int:
        return max(map(get("steps"), self.episodes)) if len(self.episodes) else 0

    @contextmanager
    def episode(self, epsilon: float) -> "MetricLoggerPD":
        start_time = time.time()
        try:
            yield self
        except Exception as e:
            raise e

        steps = max(map(len, self.curr_ep.values()))
        self.episodes.append(
            {
                "timestamp": pd.Timestamp.utcnow(),
                "episode": len(self.episodes),
                "steps": self.total_steps() + steps,
                "episode_steps": steps,
                "epsilon": epsilon,
                "time_delta": time.time() - start_time,
                **{k: self.agg_func[k](v) for k, v in self.curr_ep.items()},
            }
        )

        mapper = {
            pd.Timestamp: lambda x: x.strftime("%Y-%m-%d %H:%M:%S"),
            **{
                t: lambda x: round(x, 2)
                for t in [float, np.float16, np.float32, np.float64]
            },
        }
        print(
            {
                k: mapper.get(type(v), lambda x: x)(v)
                for k, v in sorted(self.episodes[-1].items(), key=get(0))
            }
        )

        self.curr_ep.clear()

    def record(self):
        df = (
            pd.DataFrame(self.episodes)
            # .assign(**{
            #    k: lambda x: x[k].ewm(100).mean()
            #    for k in self.agg_func.keys()
            # })
            .set_index("timestamp")
            .round(4)
        )

        df.to_csv(self.path / "log.csv")
