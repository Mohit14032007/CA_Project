#include <iostream>
#include <cassert>
#include <vector>
#include "hardware/device.h"

using namespace llm_system;

int main() {
    setenv("DEBUG_CACHE_CAPACITY", "100", 1);
    Device* device = new Device();
    device->cache_capacity_bytes = 100;

    std::cout << "Test 1: Equal-size insertion, hit promotion, LRU eviction\n";
    device->reset_expert_cache();
    device->mark_expert_resident(1, 1, 40);
    device->mark_expert_resident(1, 2, 40);
    assert(device->resident_bytes == 80);
    device->is_expert_resident(1, 1);
    device->mark_expert_resident(1, 3, 40);
    assert(device->resident_bytes == 80);
    // (1,2) should be evicted
    assert(!device->is_expert_resident(1, 2));
    assert(device->is_expert_resident(1, 1));
    assert(device->is_expert_resident(1, 3));

    std::cout << "Test 2: Variable-size eviction (Testing bug fix)\n";
    device->reset_expert_cache();
    device->mark_expert_resident(1, 1, 40);
    device->mark_expert_resident(1, 2, 40);
    // Insert C size 60 -> Evicts A (40). Bytes = 80 - 40 + 60 = 100.
    device->mark_expert_resident(1, 3, 60);
    assert(device->resident_bytes == 100);
    assert(!device->is_expert_resident(1, 1));

    std::cout << "Test 3: Multiple evictions for one insertion\n";
    device->reset_expert_cache();
    device->mark_expert_resident(1, 1, 20);
    device->mark_expert_resident(1, 2, 20);
    device->mark_expert_resident(1, 3, 20);
    assert(device->resident_bytes == 60);
    device->mark_expert_resident(1, 4, 80);
    // Evicts A, B, C.
    assert(device->resident_bytes == 80);
    assert(device->lru_expert_list.size() == 1);
    assert(!device->is_expert_resident(1, 1));

    std::cout << "Test 4: Entry larger than entire capacity\n";
    device->reset_expert_cache();
    device->mark_expert_resident(1, 1, 150);
    assert(device->resident_bytes == 150);
    assert(device->lru_expert_list.size() == 1);

    std::cout << "Test 5: Duplicate insertion\n";
    device->reset_expert_cache();
    device->mark_expert_resident(1, 1, 40);
    device->mark_expert_resident(1, 1, 40);
    assert(device->resident_bytes == 40);
    assert(device->lru_expert_list.size() == 1);

    std::cout << "Test 6: 908 complete DeepSeek-V3 experts budget\n";
    device->reset_expert_cache();
    device->cache_capacity_bytes = 80000000000LL;
    for (int i = 0; i < 908; i++) {
        device->mark_expert_resident(1, i, 88080384LL);
    }
    assert(device->resident_bytes == 79976988672LL);
    assert(device->lru_expert_list.size() == 908);
    // 909th expert
    device->mark_expert_resident(1, 908, 88080384LL);
    // Should evict (1, 0)
    assert(device->resident_bytes == 79976988672LL);
    assert(!device->is_expert_resident(1, 0));
    assert(device->is_expert_resident(1, 908));

    std::cout << "All LRU tests passed.\n";

    return 0;
}
