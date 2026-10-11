# Experiment Matrix (A100 Replication)

| Experiment | Processor | PE | ET | TP | DP | Model | Precision | Devices | Memory/device | Purpose |
|------------|-----------|----|----|----|----|-------|-----------|---------|---------------|---------|
| **A100-GPU** | GPU | OFF | 1 | 4 | 1 | Mixtral 47B | FP16 | 4 | 80 GB | A100 reference baseline |
| **A100-Duplex** | GPU+LOGIC | OFF | 1 | 4 | 1 | Mixtral 47B | FP16 | 4 | 80 GB | Evaluate Logic-PIM effect on A100 |
| **A100-Duplex+PE** | GPU+LOGIC | ON | 1 | 4 | 1 | Mixtral 47B | FP16 | 4 | 80 GB | Evaluate dynamic balancer on A100 |
| **A100-Duplex+PE+ET** | GPU+LOGIC | ON | 4 | 4 | 1 | Mixtral 47B | FP16 | 4 | 80 GB | Evaluate Expert Tensor Parallelism on A100 |
