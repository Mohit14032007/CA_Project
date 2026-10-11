#include <iostream>
#include <cassert>
#include <vector>
#include <unordered_set>
#include "dram/mmap_controller.h"
#include "dram/memory_config.h"

using namespace llm_system;

int main() {
    std::cout << "============================================================\n";
    std::cout << "Test: Production GPU DDR5 Address Mapping Validation\n";
    std::cout << "============================================================\n";

    MemoryConfig config = ddr5_32gb_x16_2ch;
    auto mmap = MMapController::Create(config);

    long long granul = config.granul;
    long long num_cube = config.num_cube;
    long long num_channel = config.num_channel;
    long long num_rank = config.num_rank;
    long long num_bg = config.num_bankgroup;
    long long num_bank = config.num_bank;
    long long num_col = config.num_col;
    long long num_row = config.num_row;

    long long row_step = granul * num_cube * num_channel * num_rank * num_bg * num_bank * num_col;
    long long total_capacity = row_step * num_row;

    std::cout << "DDR5 Config: Ch=" << num_channel << ", Rank=" << num_rank
              << ", BG=" << num_bg << ", Bank=" << num_bank
              << ", Col=" << num_col << ", Rows=" << num_row
              << ", Granularity=" << granul << " B\n";
    std::cout << "Row Step Size: " << row_step << " bytes (262,144 bytes)\n";
    std::cout << "Total Capacity: " << total_capacity << " bytes (~2.748 TB)\n\n";

    // 1. Test Address 0
    Ramulator::AddrVec_t vec0 = mmap->getAddrVec(0, 0);
    std::cout << "Addr 0 -> [Ch=" << vec0[0] << ", Rk=" << vec0[1]
              << ", BG=" << vec0[2] << ", Bk=" << vec0[3]
              << ", Row=" << vec0[4] << ", Col=" << vec0[5] << "]\n";
    assert(vec0[0] == 0 && vec0[1] == 0 && vec0[2] == 0 && vec0[3] == 0 && vec0[4] == 0 && vec0[5] == 0);

    // 2. Test Row 0 boundary (last byte in row 0)
    Ramulator::AddrVec_t vec_row0_end = mmap->getAddrVec(row_step - granul, 0);
    assert(vec_row0_end[4] == 0); // Row index should still be 0

    // 3. Test Row 1 start
    Ramulator::AddrVec_t vec_row1_start = mmap->getAddrVec(row_step, 0);
    std::cout << "Addr " << row_step << " -> Row=" << vec_row1_start[4] << "\n";
    assert(vec_row1_start[4] == 1); // Row index should be 1

    // 4. Test Expert 1 start and Expert 2 start
    long long expert_size = 88080384LL;
    Ramulator::AddrVec_t vec_exp0 = mmap->getAddrVec(0, 0);
    Ramulator::AddrVec_t vec_exp1 = mmap->getAddrVec(expert_size, 0);
    assert(vec_exp1[4] == expert_size / row_step);
    std::cout << "Expert 0 Start: Row=" << vec_exp0[4] << "\n";
    std::cout << "Expert 1 Start: Row=" << vec_exp1[4] << "\n";

    // 5. Test DeepSeek-V3 Maximum Routed Expert Address (14,592 experts)
    long long max_expert_addr = 14592LL * expert_size; // 1,285,268,963,328 B (~1.285 TB)
    Ramulator::AddrVec_t vec_max = mmap->getAddrVec(max_expert_addr, 0);
    std::cout << "Max Expert Addr (" << max_expert_addr << ") -> Row=" << vec_max[4] << "\n";
    assert(vec_max[4] == 4902912);
    assert(vec_max[4] < num_row);
    double row_util = (double)vec_max[4] / (double)num_row * 100.0;
    std::cout << "Row Capacity Utilization: " << row_util << "% (< 100% capacity)\n";

    // 6. Test Injectivity (Collision Freedom) over dense multi-expert sample
    std::cout << "Testing injectivity over sample of 10,000 addresses...\n";
    std::unordered_set<std::string> seen_coords;
    for (int e = 0; e < 100; e++) {
        long long base = (long long)e * expert_size;
        for (int b = 0; b < 100; b++) {
            long long addr = base + b * granul;
            Ramulator::AddrVec_t v = mmap->getAddrVec(addr, 0);
            std::string key = std::to_string(v[0]) + "_" + std::to_string(v[1]) + "_" +
                              std::to_string(v[2]) + "_" + std::to_string(v[3]) + "_" +
                              std::to_string(v[4]) + "_" + std::to_string(v[5]);
            assert(seen_coords.find(key) == seen_coords.end());
            seen_coords.insert(key);
        }
    }
    std::cout << "Injectivity test passed: 10,000 unique coordinates generated with 0% collisions.\n";

    std::cout << "\nALL PRODUCTION GPU DDR5 MAPPING TESTS PASSED!\n";
    return 0;
}
