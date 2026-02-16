import time

from hafnia.experiment.hafnia_logger import HafniaLogger

logger = HafniaLogger()
logger.log_scalar("train/loss", 0.1, step=1)
logger.log_scalar("train/loss", 0.2, step=2)
logger.log_scalar("train/loss", 0.3, step=3)
logger.log_scalar("train/loss", 0.4, step=3)
logger.log_scalar("train/loss", 0.5, step=5)

tdiffs = []
t0 = time.time()
[logger.log_scalar("train/loss", step * 0.1, step=step) for step in range(1000)]
t1 = time.time()
tdiffs.append(t1 - t0)
print(f"Logging 1000 scalars took {t1 - t0:.2f} seconds")


t0 = time.time()
[logger.log_scalar("train/loss", step * 0.1, step=step) for step in range(1000)]
t1 = time.time()
tdiffs.append(t1 - t0)
print(f"Logging 1000 scalars took {t1 - t0:.2f} seconds")


t0 = time.time()
[logger.log_scalar("train/loss", step * 0.1, step=step) for step in range(1000)]
t1 = time.time()
tdiffs.append(t1 - t0)
print(f"Logging 1000 scalars took {t1 - t0:.2f} seconds")


t0 = time.time()
[logger.log_scalar("train/loss", step * 0.1, step=step) for step in range(1000)]
t1 = time.time()
tdiffs.append(t1 - t0)
print(f"Logging 1000 scalars took {t1 - t0:.2f} seconds")

logger.log_writer.close()
print("asdf")
