# Machine-Readable Completion Status Format

**File:** `raw/ds5c10_completion_status.txt`  
**Format:** Key-Value plaintext (`KEY=VALUE`)  

---

## Fields Description

| Key | Type | Description |
| :--- | :--- | :--- |
| `STATUS` | string | `SUCCESS` or `FAILED (<REASON>)` |
| `EXIT_CODE` | int | Process return code (0 for success) |
| `STAGE` | string | `DeepSeek-V3 Full Baseline` |
| `MOE_LAYERS` | int | Total active MoE layers simulated (57) |
| `TOTAL_LAYERS` | int | Total transformer layers simulated (60) |
| `EXPECTED_COLD_EXPERTS`| int | Expected cold expert count ($57 \times 8 = 456$) |
| `EXPECTED_RESIDENT_BYTES`| long | Expected total resident bytes ($456 \times 88,080,384$) |
| `HOST_WALLCLOCK_SECONDS`| float | Total host runtime across both passes |
| `PASS1_HOST_SECONDS` | float | Host runtime for Pass 1 (Cold Start) |
| `PASS1_SIMULATED_DDR5_NS`| float | Total cycle-accurate DDR5 transfer time in nanoseconds |
| `PASS1_SIMULATED_DDR5_MS`| float | Total cycle-accurate DDR5 transfer time in milliseconds |
| `PASS2_HOST_SECONDS` | float | Host runtime for Pass 2 (Warm Replay) |
| `PASS2_DDR5_DELTA_NS`| float | DDR5 clock advance during Pass 2 (0 ns expected) |
| `PASS2_HBM_DELTA_NS` | float | HBM clock advance during Pass 2 |
| `RESIDENT_EXPERTS_COUNT`| int | Final number of resident experts in HBM cache |
| `RESIDENT_BYTES` | long | Final resident bytes in HBM cache |
| `ROUTING_RECORDS_PASS1`| int | Number of routing records in Pass 1 |
| `LOAD_RECORDS_PASS1` | int | Number of load telemetry records in Pass 1 |
| `CACHE_RECORDS_PASS1`| int | Number of cache records in Pass 1 |
| `FINAL_ROUTING_RECORDS`| int | Total routing records across both passes |
| `FINAL_LOAD_RECORDS` | int | Total load telemetry records across both passes |
| `FINAL_CACHE_RECORDS`| int | Total cache telemetry records across both passes |
