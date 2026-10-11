# Source-Change Provenance Manifest

**Experiment:** DeepSeek-V3 Full Baseline  
**Base Commit:** `f75d362ae157f81ef2d3bd4055c9b22e34882a9f`  
**Date:** October 2026  

---

## 1. Overview of Remediations

All modifications to the C++ simulator source code were strictly constrained to address verified correctness blockers identified during the DS-5C-9 audit. No speculative or unrelated modifications were made.

---

## 2. Detailed Change Diffs & Rationale

### 1. `src/module/linear.cpp`
**Problem:** In `Linear::forward`, `Device::is_expert_resident` and `mark_expert_resident` were previously called unconditionally for all 256 experts per layer, even when `input->getSize() == 0` (unrouted expert). This falsely populated the HBM cache with 512 unrouted experts during Pass 1 before any weight load occurred.  
**Fix:** Added an explicit token activation guard `input->getSize() > 0` before checking cache residency. Removed premature `mark_expert_resident` entirely from `Linear::forward`.

```diff
--- a/src/module/linear.cpp
+++ b/src/module/linear.cpp
@@ -45,19 +45,13 @@ Tensor::Ptr Linear::forward(const Tensor::Ptr input,
 
   A->weight_target = MemoryTarget::HBM;
   A->load_sisters = false;
-  if (A->is_expert_weight) {
+  if (A->is_expert_weight && input->getSize() > 0) {
     int layer_id = sequences_metadata->cur_layer;
     A->expert_layer_id = layer_id;
     if (!device->config.offchip_dram_cfg_path.empty()) {
       if (!device->is_expert_resident(layer_id, A->expert_id)) {
         A->weight_target = MemoryTarget::OFFCHIP_DRAM;
         A->load_sisters = true;
-        
-        long long total_expert_size = A->getSize();
-        for (auto sister : A->sister_weights) {
-            total_expert_size += sister->getSize();
-        }
-        device->mark_expert_resident(layer_id, A->expert_id, total_expert_size);
       }
     }
   }
```

---

### 2. `src/hardware/linear_impl.cpp`
**Problem:** Cold routed experts were marked resident before the load finished; sister weights did not receive the correct `expert_layer_id`; and individual matrix identities were lost in the load telemetry.  
**Fix:** Propagate `expert_layer_id` to sister weights before batching; call `device->mark_expert_resident` strictly upon successful return of `issueRamulatorBatch`; log constituent tensor rows with explicit names and durations.

```diff
--- a/src/hardware/linear_impl.cpp
+++ b/src/hardware/linear_impl.cpp
@@ -45,24 +45,30 @@ ExecStatus LinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                        DRAMRequestType::kRead, PIMOperandType::kDRAM, input, MemoryTarget::HBM);
 
     time_ns start_time = device->get_time();
-    ExecStatus weight_status =
-        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
-                       DRAMRequestType::kRead, PIMOperandType::kDRAM, weight, weight->weight_target);
-    exec_status += weight_status;
-    time_ns weight_memory_cost = weight_status.memory_duration;
-    
+    std::vector<Tensor_Ptr> batch_tensors;
+    batch_tensors.push_back(weight);
     if (weight->load_sisters) {
       for (auto sister : weight->sister_weights) {
-        ExecStatus sister_status =
-            issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
-                           DRAMRequestType::kRead, PIMOperandType::kDRAM, sister, MemoryTarget::OFFCHIP_DRAM);
-        exec_status += sister_status;
-        weight_memory_cost += sister_status.memory_duration;
+        sister->expert_layer_id = weight->expert_layer_id;
+        batch_tensors.push_back(sister);
       }
       weight->load_sisters = false;
     }
 
+    ExecStatus weight_status =
+        issueRamulatorBatch(device, LayerType::LINEAR, ProcessorType::GPU,
+                            DRAMRequestType::kRead, PIMOperandType::kDRAM, batch_tensors, weight->weight_target);
+    exec_status += weight_status;
+    time_ns weight_memory_cost = weight_status.memory_duration;
+
     if (weight->is_expert_weight) {
+      if (weight->weight_target == MemoryTarget::OFFCHIP_DRAM) {
+        long long total_expert_size = 0;
+        for (auto t : batch_tensors) {
+          total_expert_size += t->getSize();
+        }
+        device->mark_expert_resident(weight->expert_layer_id, weight->expert_id, total_expert_size);
+      }
```

---

### 3. `src/module/expert.cpp`
**Problem:** The three weight matrices within an expert were generically named `A`.  
**Fix:** Assigned unambiguous functional matrix names: `W1_gate`, `W3_up`, and `W2_down`.

```diff
--- a/src/module/expert.cpp
+++ b/src/module/expert.cpp
@@ -101,6 +101,10 @@ ExpertFFN::ExpertFFN(std::string& prefix, std::string& name,
       else if (down) w2 = down->get_tensor("A");
 
       if (w1 && w3 && w2) {
+        w1->name = "W1_gate";
+        w3->name = "W3_up";
+        w2->name = "W2_down";
+
         w1->is_expert_weight = true;
         w1->expert_id = expert_id;
         w3->is_expert_weight = true;
```

---

### 4. `src/scheduler/sequence.cpp` & `src/scheduler/sequence.h`
**Problem:** `getRandomExpert` used a static PRNG that advanced monotonically across passes, causing Pass 2 to route different expert IDs than Pass 1.  
**Fix:** Added a `cached_routing` cache in `BatchedSequence` keyed by `(token_id, cur_layer)` to deterministically replay the exact same expert selections on subsequent passes. Added `routing_file.flush()`.

```diff
--- a/src/scheduler/sequence.h
+++ b/src/scheduler/sequence.h
@@ -124,6 +124,7 @@ class BatchedSequence {
   std::vector<Sequence::Ptr> sequence;
   std::vector<int> seq_ids;
   Scheduler_ptr scheduler;
+  std::map<std::pair<int, int>, std::vector<int>> cached_routing;
 };

--- a/src/scheduler/sequence.cpp
+++ b/src/scheduler/sequence.cpp
@@ -205,15 +205,26 @@ void BatchedSequence::update_expert(int num_expert, int top_k,
...
+        std::vector<int> expert_vec;
+        auto route_key = std::make_pair(token_id, cur_layer);
+        auto it = cached_routing.find(route_key);
+        if (it != cached_routing.end()) {
+          expert_vec = it->second;
+        } else {
+          std::set<int> expert_list;
+          if(scheduler->model_config.skewness > 0){
+            expert_list = scheduler->getZipfianRandomExpert(skewness_weight, top_k);
+          }
+          else{
+            expert_list = scheduler->getRandomExpert(top_k);
+          }
+          for (auto idx : expert_list) {
+            expert_vec.push_back(idx);
+          }
+          cached_routing[route_key] = expert_vec;
+        }
```
